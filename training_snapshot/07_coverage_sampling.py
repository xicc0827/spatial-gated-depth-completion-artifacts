"""Spatial sampling recipe control, not a new algorithm."""
import torch


def sample_support(depth, count, seed, pattern):
    if depth.device.type != "cpu" or depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError("CPU 1,H,W required")
    h,w=depth.shape[-2:]
    support=torch.ones_like(depth,dtype=torch.bool)
    if pattern=="left": support[...,w//2:]=False
    elif pattern=="right": support[...,:w//2]=False
    elif pattern=="top": support[...,h//2:,:]=False
    elif pattern=="bottom": support[...,:h//2,:]=False
    elif pattern=="center_hole": support[...,h//4:3*h//4,w//4:3*w//4]=False
    elif pattern!="uniform": raise ValueError(pattern)
    valid=support & torch.isfinite(depth) & (depth>0) & (depth<10)
    ids=valid.flatten().nonzero().flatten()
    if count<=0 or len(ids)<count: raise ValueError("Insufficient valid points")
    generator=torch.Generator().manual_seed(seed)
    ids=ids[torch.randperm(len(ids),generator=generator)[:count]]
    result=torch.zeros_like(depth).flatten()
    result[ids]=depth.flatten()[ids]
    return result.reshape_as(depth)


def training_pattern(index, mixed):
    generator=torch.Generator().manual_seed(170421+index)
    slot=int(torch.randint(4,(),generator=generator))
    return ("uniform","uniform","left","right")[slot] if mixed else "uniform"


def self_test():
    depth=torch.ones(1,40,60)
    rng=torch.get_rng_state().clone()
    for name in ("uniform","left","right","top","bottom","center_hole"):
        x=sample_support(depth,500,123,name)
        assert torch.count_nonzero(x)==500 and torch.equal(x,sample_support(depth,500,123,name))
        if name=="left": assert not x[...,30:].any()
        if name=="right": assert not x[...,:30].any()
        if name=="top": assert not x[...,20:,:].any()
        if name=="bottom": assert not x[...,:20,:].any()
        if name=="center_hole": assert not x[...,10:30,15:45].any()
    assert torch.equal(rng,torch.get_rng_state())
    assert {training_pattern(i,True) for i in range(100)}=={"uniform","left","right"}
    print("PASS: exact count, deterministic masks, support constraints, RNG isolation, training shape separation")


if __name__=="__main__": self_test()
