"""Freeze second-backbone experiment before additional training or prediction."""
from pathlib import Path
import json
from datetime import datetime,timezone
from factorial_core import ARMS,state_digest
from completionformer_factorial_core import fresh
from run_factorial_controls import sha

WORK=Path(__file__).resolve().parent


def main():
    out=WORK/"completionformer_factorial";out.mkdir(exist_ok=True)
    if (out/"protocol_frozen.json").exists():raise RuntimeError("Already frozen")
    source=Path("${MODEL_ROOT}/external_baselines/CompletionFormer")
    checkpoint=Path("${MODEL_ROOT}/external_baselines/official_weights/NYUv2.pt")
    shared=WORK/"factorial_controls/protocol_frozen.json"
    hashes={name:sha(source/f"src/model/{name}.py") for name in
            ("pvt","backbone","completionformer","nlspn_module","resnet_cbam")}
    initial={str(seed):state_digest(fresh(seed).state_dict()) for seed in (42,43,44)}
    params=sum(p.numel() for p in fresh(42).parameters())
    protocol=dict(status="SECOND_BACKBONE_FACTORIAL_PREPARED_NOT_TRAINED",
        created_utc=datetime.now(timezone.utc).isoformat(),backbone="CompletionFormer",
        published_checkpoint=str(checkpoint),published_checkpoint_sha256=sha(checkpoint),
        upstream_source=str(source),upstream_source_sha256=hashes,
        upstream_commit="2744eddee9b57595dc3064f7d342569736a6803b",
        parent_provenance="Published NYUv2 weights strict-loaded through audited torchvision-DCN port; same downloaded checkpoint as earlier project studies. Not newly trained from random initialization; original training history disclosed separately.",
        shared_source_protocol=str(shared),shared_source_protocol_sha256=sha(shared),
        arms={arm:dict(teacher=flags[0],noop=flags[1]) for arm,flags in ARMS.items()},
        generic_residual="Same tensor schema/parameter count; duplicate raw normalized Z replaces the low(Z) input channel, remove output blur. This isolates the fixed-filter intervention with capacity and input information matched.",
        interface="Frozen backbone outputs pred_init_residual, guidance, confidence; add input sparse depth to pred_init; change only pred_init by sigma_P*delta; keep guidance/confidence/sparse/RGB from the same view through frozen six-step NLSPN",
        initialization="Zero final convolution gives exact pred_init endpoint; all arms share tensors per seed",
        initial_corrector_sha256=initial,trainable_parameters=params,
        within_backbone_parameter_matching=True,
        cross_backbone_parameter_matching=False,
        cross_backbone_budget_note="CF pre-NLSPN single-channel corrector uses archived hidden32 architecture; main model corrects 256 channels. Match controls within each backbone and disclose the different capacities.",
        adaptation_seeds=[42,43,44],updates=2000,effective_source_batch=2,
        task_supervision="Both uniform and patterned final decoded/propagated depths at every update, same masked L1 + r1/r2/r4 loggeometry as primary factorial",
        feature_loss="0.5 SmoothL1 beta1 teacher, 0.5 SmoothL1 beta1 no-op; the old CF screen used MSE and is a separate prior recipe",
        calibration="Four train batches at shared zero output; teacher and geometry gradient norms each target 0.5 depth norm; zero no-op gradient uses shared teacher coefficient; no validation",
        primary_lr=1e-4,weight_decay=1e-4,grad_clip=1.,alpha=1.,
        selection="Fixed final update2000 and common lr; no test/ETH3D/heldout-topology selection",
        rgb_normalization="ImageNet mean/std for CF backbone; corrector observes original unit RGB",
        standardization_epsilon=1e-6,
        historical_scope="New matched-objective CF study, not a completed replication of old B0/B1/B2 and not evidence that the old uniform-preservation failure was resolved",
        operational_gate="Wait for current main-parent GPU training to finish before GPU preflight and serialized jobs; CF parent weights remain the published checkpoint",
        code_sha256={name:sha(WORK/name) for name in
            ("completionformer_factorial_core.py","run_completionformer_factorial.py",
             "factorial_core.py","run_factorial_controls.py")},
        port_sha256={name:sha(WORK/"source_snapshot"/name) for name in
            ("completionformer_port.py","completionformer_dcn_adapter.py")},
        gpu_execution_started=False,results_available=False)
    (out/"protocol_frozen.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    print(json.dumps({"status":protocol["status"],"parameters":params,"parent_sha256":protocol["published_checkpoint_sha256"]}),flush=True)


if __name__=="__main__":main()
