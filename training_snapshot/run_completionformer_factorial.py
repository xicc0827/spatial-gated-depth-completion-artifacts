"""Second-backbone matched factorial; fixed final updates, no test selection."""
from pathlib import Path
import argparse, csv, itertools, json, random, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from factorial_core import ARMS, objective, state_digest
from completionformer_factorial_core import fresh, encode, components, calibrate_cf
from run_factorial_controls import PairedManifestDataset, sha, save_atomic

WORK=Path(__file__).resolve().parent
OUT=WORK/"completionformer_factorial"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    ap.add_argument("--arm",choices=tuple(ARMS)+("generic_residual",),required=True)
    ap.add_argument("--preflight",action="store_true")
    args=ap.parse_args();torch.set_num_threads(2)
    protocol_path=OUT/"protocol_frozen.json"
    protocol=json.loads(protocol_path.read_text())
    for name,expected in protocol["code_sha256"].items():assert sha(WORK/name)==expected,name
    for name,expected in protocol["port_sha256"].items():assert sha(WORK/"source_snapshot"/name)==expected,name
    assert sha(protocol["shared_source_protocol"])==protocol["shared_source_protocol_sha256"]
    shared=json.loads(Path(protocol["shared_source_protocol"]).read_text())
    assert sha(shared["train_split"])==shared["train_split_sha256"]
    # Scheduling amendment: the local parent stopped after migration to another host.
    if not (WORK/"clean_parent_complete.json").exists():
        amendment=json.loads((WORK/"scheduling_amendment_20261012/amendment.json").read_text())
        assert amendment["status"]=="CF_FIRST_OPERATIONAL_AMENDMENT_FROZEN"
        handoff=WORK/"migration_source_handoff_checkpoint.json"
        assert sha(handoff)==amendment["source_handoff_sha256"]
        assert json.loads(handoff.read_text())["source_training_stopped"] is True
    parent_sha=protocol["published_checkpoint_sha256"]
    assert sha(protocol["published_checkpoint"])==parent_sha
    arm_dir=OUT/f"{args.arm}_seed{args.seed}";preflight_path=arm_dir/"preflight.json"
    if not args.preflight:
        if not preflight_path.exists():raise RuntimeError("Real-source preflight required")
        report=json.loads(preflight_path.read_text())
        assert report["status"]=="REAL_SOURCE_CF_BATCH_UPDATE_PASS"
        assert report["protocol_sha256"]==sha(protocol_path) and report["parent_sha256"]==parent_sha
    sys.path[:0]=[str(WORK/"source_snapshot"),str(WORK/"source_snapshot/cross_modal_mamba_depth")]
    from nyu_pil_protocol import NYUPILSharedDataset
    from coverage_sampling import sample_support
    from completionformer_port import build_uninitialized,load_pretrained
    manifest=shared["source_manifest"][str(args.seed)]
    assert sha(manifest["path"])==manifest["sha256"]
    rows=list(csv.DictReader(Path(manifest["path"]).open(encoding="utf-8",newline="")))
    names=[r["filename"] for r in json.loads(Path(shared["train_split"]).read_text())["train"]]
    assert len(rows)==4000 and all(names[int(r["source_index"])]==r["filename"] for r in rows)
    paths=[str(Path("${NYUV2_ROOT}")/name) for name in names]
    base=NYUPILSharedDataset(paths,num_samples=500,use_aug=True,max_depth=10.)
    ds=PairedManifestDataset(base,rows,sample_support)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device("cuda")
    model=build_uninitialized(protocol["upstream_source"],prop_time=6)
    load_pretrained(model,protocol["published_checkpoint"])
    model.requires_grad_(False);model.to(device).eval()
    def encoded(batch):
        rgb,gt,su,sp=(batch[k].to(device) for k in
                    ("rgb","depth_gt","sparse_uniform","sparse_pattern"))
        return encode(model,rgb,su),encode(model,rgb,sp),gt
    calibration_path=OUT/f"calibration_seed{args.seed}.json"
    if calibration_path.exists():
        record=json.loads(calibration_path.read_text())
        assert record["parent_sha256"]==parent_sha and record["protocol_sha256"]==sha(protocol_path)
        weights=record["weights"]
    else:
        calibrator=fresh(args.seed).to(device)
        loader=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0,
                          generator=torch.Generator().manual_seed(8081))
        weights=calibrate_cf(model,calibrator,(encoded(b) for b in itertools.islice(loader,4)))
        record=dict(weights=weights,parent_sha256=parent_sha,protocol_sha256=sha(protocol_path),
            calibration_source_frames=8,backbone_batch_forwards=8,propagation_batch_forwards=8,
            backbone_frame_views=16,propagation_frame_views=16)
        calibration_path.write_text(json.dumps(record,indent=2),encoding="utf-8")
        del calibrator
    corrector=fresh(args.seed,low_band=args.arm!="generic_residual").to(device)
    assert state_digest(corrector.state_dict())==protocol["initial_corrector_sha256"][str(args.seed)]
    optimizer=torch.optim.AdamW(corrector.parameters(),lr=1e-4,weight_decay=1e-4)
    arm_dir.mkdir(exist_ok=True);checkpoint=arm_dir/"latest.pt"
    applied,seconds_before=0,0.
    if checkpoint.exists() and not args.preflight:
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        assert state["protocol_sha256"]==sha(protocol_path) and state["parent_sha256"]==parent_sha
        assert state["calibration_sha256"]==sha(calibration_path)
        assert state["source_manifest_sha256"]==manifest["sha256"]
        assert state["arm"]==args.arm and state["seed"]==args.seed
        corrector.load_state_dict(state["corrector"],strict=True);optimizer.load_state_dict(state["optimizer"])
        applied,seconds_before=state["applied"],state["training_seconds"]
        torch.set_rng_state(state["cpu_rng"]);torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    loader=DataLoader(torch.utils.data.Subset(ds,range(applied*2,len(ds))),batch_size=2,
                      shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(8081))
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        uniform,patterned,gt=encoded(batch)
        parts=components(model,corrector,uniform,patterned,gt)
        loss=objective(parts,args.arm,weights);assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(corrector.parameters(),1.)
        assert torch.isfinite(norm);optimizer.step();applied+=1
        assert all(p.grad is None for p in model.parameters())
        if args.preflight:
            assert all(torch.isfinite(p).all() for p in corrector.parameters())
            assert corrector.net[-1].weight.abs().sum()>0
            report=dict(status="REAL_SOURCE_CF_BATCH_UPDATE_PASS",arm=args.arm,seed=args.seed,
                loss=float(loss.detach()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                model_parameters_frozen=True,formal_checkpoint_modified=False,
                protocol_sha256=sha(protocol_path),parent_sha256=parent_sha)
            preflight_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps(report),flush=True);return
        if applied%20==0:print(json.dumps(dict(arm=args.arm,seed=args.seed,applied=applied,
            loss=float(loss.detach()),elapsed=time.monotonic()-start)),flush=True)
        if applied%100==0 or applied==2000:
            save_atomic(dict(corrector=corrector.state_dict(),optimizer=optimizer.state_dict(),
                arm=args.arm,seed=args.seed,applied=applied,protocol_sha256=sha(protocol_path),
                parent_sha256=parent_sha,source_manifest_sha256=manifest["sha256"],
                calibration_sha256=sha(calibration_path),weights=weights,
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                training_seconds=seconds_before+time.monotonic()-start,
                trainable_parameters=sum(p.numel() for p in corrector.parameters()),
                source_frames=applied*2,backbone_batch_forwards=applied*2,
                propagation_batch_forwards=applied*2,backbone_frame_views=applied*4,
                propagation_frame_views=applied*4,peak_gpu_bytes=torch.cuda.max_memory_allocated()),checkpoint)
        del parts,loss,uniform,patterned
    assert applied==2000
    (arm_dir/"complete.json").write_text(json.dumps(dict(status="FIXED_2000_UPDATES_COMPLETE_NOT_EVALUATED",
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),arm=args.arm,seed=args.seed,
        protocol_sha256=sha(protocol_path),parent_sha256=parent_sha,
        source_manifest_sha256=manifest["sha256"],model_inference_on_eth3d=False),indent=2),encoding="utf-8")
    print(json.dumps({"status":"FIXED_2000_UPDATES_COMPLETE_NOT_EVALUATED","backbone":"CompletionFormer","arm":args.arm,"seed":args.seed}),flush=True)


if __name__=="__main__":main()
