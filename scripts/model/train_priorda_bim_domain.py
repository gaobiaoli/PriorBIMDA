#!/usr/bin/env python3
"""Full PriorDA v1.1 BIM fine-stage domain adaptation, two training stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import torch
from huggingface_hub import hf_hub_download

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data.bim_domain import BIMDomainDataset
from bim_priorda3.engine import build_loader
from bim_priorda3.models.priorda_bim_domain import PriorDABIMDomain
from bim_priorda3.models.dav2_dense4 import dense_silog_loss
from train_dav2_dense4 import evaluate
from train_bim_early_fusion_dense import atomic_json, atomic_torch_save, selected_batch, plain, write_history


def initialization_audit(model, batch):
    model.eval()
    patch = model.fine.pretrained.patch_embed
    adapted = patch.alpha_proj
    with torch.no_grad():
        actual = model(batch)["depth"]
        patch.alpha_proj = adapted.pretrained
        try:
            original = model(batch)["depth"]
        finally:
            patch.alpha_proj = adapted
    if not torch.equal(actual, original):
        raise AssertionError("Adapter is not initially identical to official fine network")
    if not bool(adapted.pretrained.weight.count_nonzero()):
        raise AssertionError("Pretrained alpha projection is unexpectedly zero")
    groups = model.configure_stage(1)
    model.train()
    output = model(batch)
    loss = dense_silog_loss(output["depth"], batch["gt_depth"], batch["gt_valid"])["total"]
    loss.backward()
    norms = {}
    for group in groups:
        grads = [p.grad for p in group["params"] if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise AssertionError(f"Missing/nonfinite gradient: {group['name']}")
        norms[group["name"]] = sum(float(g.float().square().sum()) for g in grads) ** .5
        if norms[group["name"]] == 0:
            raise AssertionError(f"Dead group: {group['name']}")
    for block in adapted.adapter:
        if block.branch[-1].weight.grad.abs().sum() == 0:
            raise AssertionError("Adapter final projection has zero gradient")
    frozen_clean = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if not frozen_clean:
        raise AssertionError("Frozen parameter received a gradient")
    model.zero_grad(set_to_none=True)
    return dict(strict_full_checkpoint_load=True, exact_official_fine_identity=True,
                max_initial_difference_m=float((actual-original).abs().max()),
                pretrained_alpha_nonzero=True, stage1_gradient_norms=norms,
                frozen_parameters_without_gradients=frozen_clean)


def run(args):
    cfg = load_config(args.config)
    out = args.output_dir or resolve_project_path(cfg, cfg.experiment.output_dir)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("receipt.json", "latest.pt", "status.json")):
        raise FileExistsError(f"Refusing to overwrite {out}")
    atomic_json(out / "status.json", dict(state="INITIALIZING", pid=os.getpid()))
    try:
        train(cfg, out, args)
    except BaseException as error:
        atomic_json(out / "status.json", dict(state="FAILED", error=repr(error), pid=os.getpid()))
        raise


def train(cfg, out, args):
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    seed = int(cfg.experiment.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    spec = cfg.model.priorda_bim_domain
    path = Path(hf_hub_download(repo_id=spec.checkpoint_repo,
        filename=spec.checkpoint_filename, revision=spec.checkpoint_revision))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec.checkpoint_sha256:
        raise ValueError(f"Official checkpoint SHA mismatch: {digest}")
    model = PriorDABIMDomain.from_checkpoint(
        resolve_project_path(cfg, spec.official_repository), path).to(device)
    if cfg.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    datasets = {split: BIMDomainDataset(cfg, split) for split in ("train", "val")}
    if args.smoke:
        for dataset in datasets.values():
            dataset.records = dataset.records[:4]
    workers = 0 if args.smoke else int(cfg.train.num_workers)
    generator = torch.Generator().manual_seed(seed)
    loaders = {split: build_loader(ds, int(cfg.train.batch_size), workers, split == "train",
        generator=generator if split == "train" else torch.Generator().manual_seed(seed+17),
        persistent_workers=False) for split, ds in datasets.items()}
    # This is an initialization/gradient check, NOT a zero-shot benchmark.
    batch = selected_batch(next(iter(loaders["val"])), device)
    audit = initialization_audit(model, batch)
    del batch
    stage1_epochs = 1 if args.smoke else int(cfg.train.stage1_epochs)
    stage2_epochs = 1 if args.smoke else int(cfg.train.stage2_epochs)
    epochs = stage1_epochs + stage2_epochs
    lr_args = dict(adapter_lr=float(cfg.train.adapter_learning_rate),
                   decoder_lr=float(cfg.train.decoder_learning_rate),
                   alpha_lr=float(cfg.train.alpha_learning_rate),
                   vit_lr=float(cfg.train.vit_learning_rate))
    optimizer = torch.optim.AdamW(model.configure_stage(1, **lr_args),
                                 weight_decay=float(cfg.train.weight_decay))
    receipt = dict(initialization=audit, checkpoint_path=str(path), checkpoint_sha256=digest,
        config=plain(dict(cfg)), stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs,
        schedule="constant requested group LRs; preserve Adam state at stage transition",
        stage2_start="latest stage1 checkpoint in memory; no optimizer reset",
        seed_formula="base_seed + zero_based_epoch * dataset_size + sample_index",
        train_frames=len(datasets["train"]), val_frames=len(datasets["val"]),
        split_provenance=datasets["train"].split_provenance,
        loss="only ZoeDepth SILog, microbatch GT-valid pixels; gradient accumulation",
        coarse_stage=False, zero_shot_evaluation=False)
    atomic_json(out / "receipt.json", receipt)
    snapshot = out / "source"
    snapshot.mkdir()
    for source in (Path(__file__), Path(args.config),
                   Path(__file__).parents[2] / "src/bim_priorda3/models/priorda_bim_domain.py",
                   Path(__file__).parents[2] / "src/bim_priorda3/data/bim_domain.py"):
        shutil.copy2(source, snapshot / source.name)
    print("INITIALIZATION " + json.dumps(audit), flush=True)
    amp = device.type == "cuda" and bool(cfg.train.amp)
    accumulation = 2 if args.smoke else int(cfg.train.gradient_accumulation)
    history, best, steps = [], float("inf"), 0
    stage_best = {1: float("inf"), 2: float("inf")}
    for epoch in range(epochs):
        stage = 1 if epoch < stage1_epochs else 2
        if epoch == stage1_epochs:
            new_groups = model.configure_stage(2, **lr_args)
            optimizer.add_param_group(new_groups[-1])
        datasets["train"].epoch = epoch
        generator.manual_seed(seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started, total, samples = time.perf_counter(), 0., 0
        kinds = [0, 0, 0, 0]
        atomic_json(out / "status.json", dict(state="TRAINING", stage=stage, epoch=epoch+1, pid=os.getpid()))
        for index, raw in enumerate(loaders["train"]):
            batch = selected_batch(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(batch)["depth"]
                loss = dense_silog_loss(pred, batch["gt_depth"], batch["gt_valid"])["total"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at {epoch+1}:{index+1}")
            # Correctly normalize a partial final accumulation window.
            window_start = index // accumulation * accumulation
            window_size = min(accumulation, len(loaders["train"]) - window_start)
            (loss / window_size).backward()
            if (index+1) % accumulation == 0 or index+1 == len(loaders["train"]):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                    float(cfg.train.gradient_clip_norm), error_if_nonfinite=True)
                if index < accumulation:
                    for group in optimizer.param_groups:
                        group_norm = sum(float(p.grad.float().square().sum()) for p in group["params"] if p.grad is not None) ** .5
                        if group_norm == 0:
                            raise RuntimeError(f"Dead training group {group['name']}")
                        print(f"GRAD stage={stage} group={group['name']} norm={group_norm:.8g}", flush=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            n = batch["rgb"].shape[0]
            total += float(loss.detach()) * n
            samples += n
            for kind in raw["augmentation_kind"].tolist():
                kinds[kind] += 1
            if index == 0 or (index+1) % 20 == 0:
                print(f"stage={stage} epoch={epoch+1}/{epochs} batch={index+1}/{len(loaders['train'])} silog={float(loss.detach()):.6f} optimizer_steps={steps}", flush=True)
        val = evaluate(model, loaders["val"], device, amp, torch.bfloat16)
        score = val["final"]["frame_macro"]["abs_rel"]
        row = dict(epoch=epoch+1, stage=stage, train_silog=total/samples,
                   val_abs_rel=score, val_rmse=val["final"]["frame_macro"]["rmse"],
                   val_delta1=val["final"]["frame_macro"]["delta1"],
                   optimizer_steps=steps, seconds=time.perf_counter()-started,
                   augmentation_counts=kinds)
        history.append(row)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1,
                       stage=stage, config=receipt["config"], history=history,
                       checkpoint_sha256=digest, torch_rng_state=torch.get_rng_state(),
                       cuda_rng_state=torch.cuda.get_rng_state_all())
        atomic_torch_save(out / "latest.pt", payload)
        if score < best:
            best = score
            atomic_torch_save(out / "best.pt", payload)
        if score < stage_best[stage]:
            stage_best[stage] = score
            atomic_torch_save(out / f"stage{stage}_best.pt", payload)
        if epoch+1 == stage1_epochs:
            atomic_torch_save(out / "stage1_latest.pt", payload)
        atomic_json(out / f"val_epoch_{epoch+1}.json", val)
        atomic_json(out / "history.json", history)
        write_history(out / "training_history.csv", history)
        print("VAL " + json.dumps(row), flush=True)
    # Only post-finetuning in-domain test. No Matterport/zero-shot pipeline.
    if not args.smoke:
        test_ds = BIMDomainDataset(cfg, "test")
        loader = build_loader(test_ds, int(cfg.train.batch_size), workers, False, persistent_workers=False)
        tests = {}
        for name in ("stage1_best", "stage2_best"):
            payload = torch.load(out / f"{name}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            checkpoint_epoch = payload["epoch"]
            del payload
            tests[name] = dict(epoch=checkpoint_epoch, metrics=evaluate(model, loader, device, amp, torch.bfloat16))
        atomic_json(out / "test_summary.json", tests)
    atomic_json(out / "status.json", dict(state="COMPLETED", epochs=epochs, optimizer_steps=steps,
        best_val_abs_rel=best, stage_best_val_abs_rel=stage_best, pid=os.getpid()))
    print("TRAIN_COMPLETE " + json.dumps(dict(best=best, stage_best=stage_best)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
