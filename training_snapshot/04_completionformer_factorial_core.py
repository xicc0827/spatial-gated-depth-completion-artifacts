"""Matched teacher/no-op port at CompletionFormer's pre-NLSPN depth interface."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from factorial_core import lowpass, masked_l1, geometry, grad_norm, state_digest


def normalize_rgb(rgb):
    mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]
    std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    return (rgb-mean)/std


def standardize(x):
    x=x.float()
    mu=x.mean((-2,-1),keepdim=True)
    sigma=((x-mu).square().mean((-2,-1),keepdim=True)+1e-6).sqrt()
    return (x-mu)/sigma,mu,sigma


class PreNLSPNCorrector(nn.Module):
    def __init__(self,low_band=True,hidden=32):
        super().__init__();self.low_band=low_band
        self.net=nn.Sequential(nn.Conv2d(7,hidden,5,padding=2,bias=False),
            nn.GroupNorm(8,hidden),nn.GELU(),
            nn.Conv2d(hidden,hidden,3,padding=1,bias=False),nn.GroupNorm(8,hidden),nn.GELU(),
            nn.Conv2d(hidden,hidden,3,padding=1,bias=False),nn.GroupNorm(8,hidden),nn.GELU(),
            nn.Conv2d(hidden,1,3,padding=1,bias=True))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,z,rgb,sparse):
        low=lowpass(z) if self.low_band else z
        x=torch.cat((z,low,rgb.float(),(sparse>0).float(),(sparse/10.).clamp(0,1)),1)
        output=self.net(x)
        return lowpass(output) if self.low_band else output


def fresh(seed,low_band=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return PreNLSPNCorrector(low_band)


def encode(model,rgb,sparse):
    rgb_norm=normalize_rgb(rgb)
    with torch.no_grad():
        init,guide,confidence=model.backbone(rgb_norm,sparse)
        init=init+sparse
    return dict(init=init.float().detach(),guide=guide.detach(),
        confidence=confidence.detach() if confidence is not None else None,
        sparse=sparse,rgb_unit=rgb,rgb_norm=rgb_norm)


def propagate(model,view,init):
    if model.prop_time>0:
        result,*_=model.prop_layer(init,view["guide"],view["confidence"],
                                  view["sparse"],view["rgb_norm"])
    else:result=init
    return result.clamp_min(0)


def predict_delta(corrector,view):
    return corrector(standardize(view["init"])[0].detach(),view["rgb_unit"],view["sparse"])


def corrected(view,delta):
    # Preserve the natural endpoint exactly; no mu+sigma*Z reconstruction.
    return view["init"]+standardize(view["init"])[2].detach()*delta


def components(model,corrector,uniform,patterned,gt):
    du,dp=predict_delta(corrector,uniform),predict_delta(corrector,patterned)
    pu=propagate(model,uniform,corrected(uniform,du))
    pp=propagate(model,patterned,corrected(patterned,dp))
    target=(lowpass(standardize(uniform["init"])[0])-
            lowpass(standardize(patterned["init"])[0])).detach()
    return dict(depth=.5*(masked_l1(pu,gt)+masked_l1(pp,gt)),
        geom=.5*(geometry(pu,gt)+geometry(pp,gt)),
        teacher=.5*F.smooth_l1_loss(dp,target,beta=1.),
        noop=.5*F.smooth_l1_loss(du,torch.zeros_like(du),beta=1.),
        target=target,delta_uniform=du,delta_pattern=dp,pred_uniform=pu,pred_pattern=pp)


def calibrate_cf(model,corrector,batches):
    norms={key:[] for key in ("depth","geom","teacher","noop")}
    before=state_digest(corrector.state_dict())
    for uniform,patterned,gt in batches:
        c=components(model,corrector,uniform,patterned,gt)
        for key in norms:norms[key].append(grad_norm(c[key],corrector.parameters()))
    means={k:float(np.mean(v)) for k,v in norms.items()}
    if any(not np.isfinite(means[k]) or means[k]<=1e-12 for k in ("depth","geom","teacher")):
        raise ValueError(f"Degenerate CF calibration: {means}")
    assert means["noop"]==0 and before==state_digest(corrector.state_dict())
    return dict(geom=.5*means["depth"]/means["geom"],feature=.5*means["depth"]/means["teacher"],
        gradient_norms=norms,mean_gradient_norms=means,initialization_sha256=before,
        policy="Same matched factorial calibration as main backbone; calibrated independently for CF units; no-op shares teacher coefficient because its initial gradient is zero")
