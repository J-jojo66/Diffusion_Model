import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weighted=1.0):
        """
        pred, targ : [batch_size, action_dim]
        """
        loss = self._loss(pred, targ)
        WeightedLoss = (weighted * loss).mean()
        return WeightedLoss

class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')
    
Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2
}

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class SinusoidalPosEmb(nn.Module):  #正余弦位置编码
    def __init__(self, dim):
        super(SinusoidalPosEmb,self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim-1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb


class MLP(nn.Module): #定义一个MLP为去噪神经网络
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16): 
        #参数传入状态空间、动作空间、隐藏层时间戳的维度
        super(MLP, self).__init__()

        self.t_dim = t_dim 
        self.a_dim = action_dim
        self.device = device

        self.time_mlp = nn.Sequential( #对时间维度进行位置编码
            SinusoidalPosEmb(t_dim), #将时间标量映射为维度为t_dim的向量
            nn.Linear(t_dim, t_dim*2), #全连接层，扩展维度，提供高维的非线性变换特征空间
            nn.Mish(), #激活函数，引入非线性
            nn.Linear(t_dim*2, t_dim)
        )

        #中间层
        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )

        #输出成
        self.final_layer = nn.Linear(hidden_dim, action_dim)

        #参数初始化
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                

    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)
    
#构建diffusion的主体
class Diffusion(nn.Module):
    def __init__(self, loss_type, beta_schedule="linear", clip_denoised=True, **kwargs):
        super().__init__()
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.T = kwargs["T"]
        self.device = torch.device(kwargs["device"])
        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, self.device)

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0) #逐次乘法累积
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]]) #

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev) #构建一个缓冲区

        #前向过程
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        #反向过程
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)) #根据xt方向求解x0用到两个超参数

        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)) #求均值的两个超参数

        self.loss_fn = Losses[loss_type]()

    #采样过程
    def q_posterior(self, x_start, x, t): #求后燕概率函数
        posterior_mean = (extract(self.posterior_mean_coef1, t, x.shape) * x_start
                          + extract(self.posterior_mean_coef2, t, x.shape) * x)
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return posterior_mean, posterior_variance, posterior_log_variance


    def predict_start_from_noise(self, x, t, predict_noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x 
                - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * predict_noise)


    def p_mean_variance(self, x, t, s):
        pred_noise = self.model(x, t, s) #预测噪声
        x_recon = self.predict_start_from_noise(x, t, pred_noise) #预测噪声的反向过程
        x_recon.clamp(-1, 1)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance
    
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, s)#计算方向扩散过程的均值和方差
        noise = torch.randn_like(x) 

        #生成mask--最后一步不需要noise
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) #解包 生成长度比x少一位的1张量
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise


    def p_sample_loop(self, state, shape, *args, **kwargs):
        device = self.device
        batch_size = state.shape[0]
        x = torch.randn(shape, device=device, requires_grad=False) #初始化随机噪声 不使用反向过程的梯度

        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state)


        return x

    def sample(self, state, *args, **kwargs):
        """
        state : [batch_size, state_dim]"""
        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim]
        action = self.p_sample_loop(state, shape, *args, **kwargs) #反向不断采样
        return action.clamp(-1, 1) #限制动作范围

    #训练
    def q_sample(self, x_start, t, noise):
        sample = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                  + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)#生成标签值
        x_recon = self.model(x_noisy, t, state)#生成模型预测值

        loss = self.loss_fn(x_recon, x_noisy, weights)
        return loss

    def loss(self, x, state, weights=1.0):#x为真实数据
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size, ), device=self.device).long() #生成随机时间戳
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    print(f"device:{device}")
    x = torch.randn(256, 2).to(device)#模拟输入，[batch_size, action_dim]
    print(f"x:{x}")
    state = torch.randn(256, 11).to(device)#模拟状态，[batch_size, state_dim]
    model = Diffusion(loss_type="l2", obs_dim=11, act_dim=2, hidden_dim=256, T=100, device=device).to(device)
    action = model(state)

    loss = model.loss(x, state)
    # print(f"action:{action}; loss:{loss.item()}")
    print(f"loss:{loss.item()}")



