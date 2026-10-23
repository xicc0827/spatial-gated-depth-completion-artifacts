"""Executable matched controls. GPU work is gated by verified fixed-72 parent."""
from pathlib import Path
import argparse, csv, hashlib, itertools, json, random, shutil, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from factorial_core import (ARMS, fresh_corrector, state_digest, isolated_augmentation,
                            encode_frozen, calibrate, loss_components, objective)

WORK = Path(__file__).resolve().parent
OUT = WORK / "factorial_controls"


def sha(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def save_atomic(payload, path):
    if shutil.disk_usage(path.parent).free < 5 * 1024 ** 3:
        raise RuntimeError("Insufficient reserved space for safe checkpoints")
    temp = path.with_suffix(".tmp.pt")
    torch.save(payload,temp); temp.replace(path)


def gate_parent(protocol):
    for name, expected in protocol["code_sha256"].items():
        assert sha(WORK / name) == expected, f"Frozen experiment code changed: {name}"
    complete_path = WORK / "clean_parent_complete.json"
    if not complete_path.exists():
        raise RuntimeError("WAITING_FOR_FIXED_EPOCH72_PARENT: no downstream GPU work started")
    complete = json.loads(complete_path.read_text())
    assert complete["status"] == "FIXED_EPOCH72_SOURCE_ONLY_PARENT_COMPLETE"
    path = Path(complete["checkpoint"])
    assert sha(path) == complete["checkpoint_sha256"]
    state = torch.load(path,map_location="cpu",weights_only=False)
    assert state["epoch"] == 72 and len(state["history"]) == 72
    assert [r["epoch"] for r in state["history"]] == list(range(1,73))
    assert state["variant"] == "adaptive" and state["seed"] == 42
    assert all(torch.isfinite(v).all() for v in state["model"].values())
    assert sha(protocol["train_split"]) == protocol["train_split_sha256"]
    snapshot_plan = json.loads((WORK / "clean_parent_resume_plan.json").read_text())
    for source, expected in snapshot_plan["snapshot_sha256"].items():
        assert sha(source) == expected, f"Snapshot changed: {source}"
    return complete, state


class PairedManifestDataset(Dataset):
    def __init__(self, base, rows, sample_support):
        self.base, self.rows, self.sample_support = base, rows, sample_support

    def __len__(self):return len(self.rows)

    def __getitem__(self, position):
        row = self.rows[position]
        with isolated_augmentation(int(row["augmentation_seed"])):
            item = self.base[int(row["source_index"])]
        gt = item["depth_gt"]
        return dict(rgb=item["rgb"], depth_gt=gt,
            sparse_uniform=self.sample_support(gt,int(row["count"]),int(row["support_seed"]),"uniform"),
            sparse_pattern=self.sample_support(gt,int(row["count"]),int(row["support_seed"]),row["pattern"]),
            position=position)


def encoded(model, batch, device):
    rgb, gt, su, sp = (batch[k].to(device) for k in
                      ("rgb","depth_gt","sparse_uniform","sparse_pattern"))
    return encode_frozen(model,rgb,su), encode_frozen(model,rgb,sp), su, sp, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    ap.add_argument("--arm",choices=tuple(ARMS)+("generic_residual",),required=True)
    ap.add_argument("--preflight",action="store_true",help="One real effective-batch update; never a formal result")
    args = ap.parse_args()
    torch.set_num_threads(2)
    protocol_path = OUT / "protocol_frozen.json"
    protocol = json.loads(protocol_path.read_text())
    complete, parent = gate_parent(protocol)  # Before CUDA initialization.
    arm_dir = OUT / f"{args.arm}_seed{args.seed}"
    preflight_path = arm_dir / "preflight.json"
    if not args.preflight:
        if not preflight_path.exists():
            raise RuntimeError("Real-source one-update preflight is required before formal training")
        preflight = json.loads(preflight_path.read_text())
        assert preflight["status"] == "REAL_SOURCE_BATCH_UPDATE_PASS"
        assert preflight["protocol_sha256"] == sha(protocol_path)
        assert preflight["parent_sha256"] == complete["checkpoint_sha256"]
    sys.path[:0] = [str(WORK / "source_snapshot"), str(WORK / "source_snapshot/cross_modal_mamba_depth")]
    from research_protocol import CrossModalMambaDepth, dataset
    from coverage_sampling import sample_support
    manifest = protocol["source_manifest"][str(args.seed)]
    assert sha(manifest["path"]) == manifest["sha256"]
    rows = list(csv.DictReader(Path(manifest["path"]).open(encoding="utf-8",newline="")))
    assert len(rows) == 4000
    names = [r["filename"] for r in json.loads(Path(protocol["train_split"]).read_text())["train"]]
    assert all(names[int(r["source_index"])] == r["filename"] for r in rows)
    base = dataset("${NYUV2_ROOT}",names,training=True,resize_backend="pil")
    ds = PairedManifestDataset(base,rows,sample_support)
    device = torch.device("cuda")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = CrossModalMambaDepth(min_depth=0.,gate_type="adaptive")
    model.load_state_dict(parent["model"],strict=True); model.requires_grad_(False)
    model.to(device).eval(); del parent
    # Source only: infer feature dimensions from the first train pair, never test.
    first = next(iter(DataLoader(ds,batch_size=2,shuffle=False,num_workers=0,
                                 generator=torch.Generator().manual_seed(8081))))
    bundle = encoded(model,first,device)
    rgb_channels, depth_channels = bundle[0][2][0].shape[1], bundle[0][2][1].shape[1]
    assert (depth_channels,rgb_channels) == (256,256)
    calibrator = fresh_corrector(args.seed,depth_channels,rgb_channels).to(device)
    assert state_digest(calibrator.state_dict()) == manifest["initial_corrector_sha256"]
    calibration_path = OUT / f"calibration_seed{args.seed}.json"
    if calibration_path.exists():
        calibration_record = json.loads(calibration_path.read_text())
        assert calibration_record["parent_sha256"] == complete["checkpoint_sha256"]
        assert calibration_record["protocol_sha256"] == sha(protocol_path)
        weights = calibration_record["weights"]
    else:
        loader = DataLoader(ds,batch_size=2,shuffle=False,num_workers=0,
                            generator=torch.Generator().manual_seed(8081))
        def batches():
            for b in itertools.islice(loader,4):
                yield encoded(model,b,device)
        weights = calibrate(model,calibrator,batches())
        calibration_record = dict(weights=weights,parent_sha256=complete["checkpoint_sha256"],
            protocol_sha256=sha(protocol_path), calibration_source_frames=8,
            calibration_encoder_batch_forwards=8, calibration_decoder_batch_forwards=8,
            calibration_encoder_frame_views=16, calibration_decoder_frame_views=16,
            dimension_probe_source_frames=2, dimension_probe_encoder_batch_forwards=2)
        calibration_path.write_text(json.dumps(calibration_record,indent=2),encoding="utf-8")
    del calibrator, bundle, first
    corrector = fresh_corrector(args.seed,depth_channels,rgb_channels,
                               low_band=args.arm != "generic_residual").to(device)
    assert state_digest(corrector.state_dict()) == manifest["initial_corrector_sha256"]
    optimizer = torch.optim.AdamW(corrector.parameters(),lr=1e-4,weight_decay=1e-4)
    arm_dir.mkdir(exist_ok=True)
    checkpoint = arm_dir / "latest.pt"
    applied, seconds_before = 0, 0.
    if checkpoint.exists() and not args.preflight:
        state = torch.load(checkpoint,map_location="cpu",weights_only=False)
        assert state["protocol_sha256"] == sha(protocol_path)
        assert state["parent_sha256"] == complete["checkpoint_sha256"]
        assert state["source_manifest_sha256"] == manifest["sha256"]
        assert state["calibration_sha256"] == sha(calibration_path)
        assert state["arm"] == args.arm and state["seed"] == args.seed
        corrector.load_state_dict(state["corrector"],strict=True)
        optimizer.load_state_dict(state["optimizer"])
        applied, seconds_before = state["applied"], state["training_seconds"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    pending_ds = torch.utils.data.Subset(ds,range(applied*2,len(ds)))
    loader = DataLoader(pending_ds,batch_size=2,shuffle=False,num_workers=0,
                        generator=torch.Generator().manual_seed(8081))
    for b in loader:
        optimizer.zero_grad(set_to_none=True)
        uniform, patterned, su, sp, gt = encoded(model,b,device)
        components = loss_components(model,corrector,uniform,patterned,su,sp,gt)
        loss = objective(components,args.arm,weights)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(corrector.parameters(),1.)
        assert torch.isfinite(norm)
        optimizer.step(); applied += 1
        assert all(p.grad is None for p in model.parameters())
        if args.preflight:
            assert all(torch.isfinite(p).all() for p in corrector.parameters())
            assert corrector.out.weight.detach().abs().max() > 0
            report=dict(status="REAL_SOURCE_BATCH_UPDATE_PASS",arm=args.arm,seed=args.seed,
                loss=float(loss.detach()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                model_parameters_frozen=True,formal_checkpoint_modified=False,
                protocol_sha256=sha(protocol_path),parent_sha256=complete["checkpoint_sha256"])
            (arm_dir/"preflight.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps(report),flush=True);return
        if applied % 20 == 0:
            print(json.dumps(dict(arm=args.arm,seed=args.seed,applied=applied,
                loss=float(loss.detach()),elapsed=time.monotonic()-start)),flush=True)
        if applied % 100 == 0 or applied == 2000:
            save_atomic(dict(corrector=corrector.state_dict(),optimizer=optimizer.state_dict(),
                arm=args.arm,seed=args.seed,applied=applied,
                protocol_sha256=sha(protocol_path),parent_sha256=complete["checkpoint_sha256"],
                source_manifest_sha256=manifest["sha256"],calibration_sha256=sha(calibration_path),
                weights=weights, cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                training_seconds=seconds_before+time.monotonic()-start,
                trainable_parameters=sum(p.numel() for p in corrector.parameters()),
                source_frames=applied*2, encoder_batch_forwards=applied*2,
                decoder_batch_forwards=applied*2,encoder_frame_views=applied*4,
                decoder_frame_views=applied*4,peak_gpu_bytes=torch.cuda.max_memory_allocated()),checkpoint)
        del components, loss, uniform, patterned
    assert applied == 2000
    (arm_dir/"complete.json").write_text(json.dumps(dict(status="FIXED_2000_UPDATES_COMPLETE_NOT_EVALUATED",
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),arm=args.arm,seed=args.seed,
        protocol_sha256=sha(protocol_path),parent_sha256=complete["checkpoint_sha256"],
        source_manifest_sha256=manifest["sha256"],model_inference_on_eth3d=False),indent=2),encoding="utf-8")
    print(json.dumps({"status":"FIXED_2000_UPDATES_COMPLETE_NOT_EVALUATED","arm":args.arm,"seed":args.seed}),flush=True)


if __name__ == "__main__":
    main()
