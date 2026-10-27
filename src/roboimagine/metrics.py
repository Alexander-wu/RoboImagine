"""Per-frame visual metrics followed by a video mean, matching archived evaluation."""
import numpy as np
import torch
class Metrics:
    def __init__(self,device):
        import lpips,piqa
        self.device=device
        self.lpips=lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False)
        self.ssim=piqa.SSIM(reduction='none').to(device).eval().requires_grad_(False)
    @torch.inference_mode()
    def __call__(self,prediction,truth):
        p=torch.as_tensor(np.ascontiguousarray(prediction),device=self.device).float().permute(0,3,1,2)
        t=torch.as_tensor(np.ascontiguousarray(truth),device=self.device).float().permute(0,3,1,2)/255
        if p.shape!=t.shape or not torch.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid prediction pixels')
        mse=(p-t).square().mean((1,2,3));mae=(p-t).abs().mean((1,2,3))
        lp=torch.cat([self.lpips(p[i:i+2]*2-1,t[i:i+2]*2-1).flatten() for i in range(0,len(p),2)])
        ss=torch.cat([self.ssim(p[i:i+2],t[i:i+2]) for i in range(0,len(p),2)])
        return {k:float(v.mean()) for k,v in {'mse':mse,'mae':mae,'psnr':-10*mse.clamp_min(1e-12).log10(),'lpips':lp,'ssim':ss}.items()}
