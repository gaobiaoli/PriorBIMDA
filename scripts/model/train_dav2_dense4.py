#!/usr/bin/env python3
"""Fresh six-epoch dense4 metric training; no scale/residual supervision."""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
import yaml
from torch.utils.data import Subset

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data.dense4 import Dense4Dataset
from bim_priorda3.data.augmentation import apply_da3_global_scale_perturbation
from bim_priorda3.data.stanford2d3ds import load_stanford_all_valid_depth, official_regular_depth_path
from bim_priorda3.early_fusion import DenseDepthMetricAccumulator
from bim_priorda3.engine import build_loader
from bim_priorda3.models.dav2_dense4 import DAv2Dense4, dense_silog_loss
from bim_priorda3.models.priorda_relative_metric_refiner import (
    PriorDARelativeMetricRefiner,
    PriorDARelativePriorFrameMetricRefiner,
    PriorDARelativePriorIdentityMetricRefiner,
    PriorDARelativeZeroAnchorMetricRefiner,
)
from train_bim_early_fusion_dense import atomic_json, atomic_torch_save, plain, resolve_checkpoint, selected_batch, write_history
from train_dav2_joint_scale_low import seed_everything


class BothMetrics:
    def __init__(self):
        self.micro = DenseDepthMetricAccumulator()
        self.frame_sum = {}
        self.frames = 0

    def update(self, pred, gt, support):
        if not bool((torch.isfinite(pred) & (pred > 0)).all()):
            raise FloatingPointError("Invalid prediction; refusing to shrink GT metric support")
        self.micro.update(pred, gt, support)
        for p, g, m in zip(pred, gt, support, strict=True):
            frame = DenseDepthMetricAccumulator()
            frame.update(p[None], g[None], m[None])
            for k, v in frame.compute().items():
                if k != "count":
                    self.frame_sum[k] = self.frame_sum.get(k, 0.) + float(v)
            self.frames += 1

    def compute(self):
        return {"pixel_micro": self.micro.compute(), "frame_macro": {k: v/self.frames for k,v in self.frame_sum.items()}, "frames": self.frames}


@torch.inference_mode()
def evaluate(model, loader, device, amp, amp_dtype=torch.float16):
    model.eval()
    final, raw = BothMetrics(), BothMetrics()
    start = time.perf_counter()
    for raw_batch in loader:
        batch = selected_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            pred = model(batch)["depth"]
        support = batch["gt_valid"].bool() & torch.isfinite(batch["gt_depth"]) & (batch["gt_depth"] > 0)
        final.update(pred, batch["gt_depth"], support)
        raw.update(batch["base_depth"], batch["gt_depth"], support)
    return {"final": final.compute(), "raw_da3": raw.compute(), "seconds": time.perf_counter()-start,
            "support": "official all finite positive GT, no depth cutoff", "alignment": "none"}


def audit_depth_range(dataset, *, policy):
    def scan(record):
        d,m = load_stanford_all_valid_depth(official_regular_depth_path(record["image"]), (dataset.height,dataset.width))
        v=d[m>0]
        return int(v.size), int((v>20).sum()), float(v.max())
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(scan,dataset.records))
    n=sum(r[0] for r in rows)
    return {"split":"train", "frames":len(rows), "gt_valid_pixels":n,
            "pixels_above_head_max20":sum(r[1] for r in rows),
            "fraction_above_head_max20":sum(r[1] for r in rows)/n,
            "frames_above_head_max20":sum(r[1]>0 for r in rows),
            "max_gt_metres":max(r[2] for r in rows),
            "policy":policy}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--results-dir",type=Path)
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--max-train-samples",type=int)
    parser.add_argument("--max-val-samples",type=int)
    parser.add_argument("--skip-test",action="store_true")
    args=parser.parse_args()
    cfg=load_config(args.config)
    seed=int(cfg.experiment.seed)
    seed_everything(seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    priorda_affine=bool(cfg.model.get("priorda_relative_metric_refiner", {}).get("enabled", False))
    priorda_zero_anchor=bool(cfg.model.get("priorda_relative_zero_anchor_refiner", {}).get("enabled", False))
    priorda_identity=bool(cfg.model.get("priorda_relative_prior_identity_refiner", {}).get("enabled", False))
    priorda_prior_frame=bool(cfg.model.get("priorda_relative_prior_frame_refiner", {}).get("enabled", False))
    if sum((priorda_affine, priorda_zero_anchor, priorda_identity, priorda_prior_frame)) > 1:
        raise ValueError("Enable exactly one PriorDA-relative metric representation")
    priorda_relative=priorda_affine or priorda_zero_anchor or priorda_identity or priorda_prior_frame
    if priorda_relative:
        aug=cfg.train.augment
        if float(aug.bim_full_dropout_probability) != 0:
            raise ValueError("BIM-defined normalization is undefined after full BIM dropout")
        if float(aug.bim_shuffle_probability) != 0:
            raise ValueError("BIM shuffle changes the output affine frame and is disabled for this control")
    if (priorda_zero_anchor or priorda_identity or priorda_prior_frame) and (
        float(cfg.train.augment.da3_global_scale_probability) != 0
        or float(cfg.train.augment.da3_global_scale_log_range) != 0
    ):
        raise ValueError("PriorDA representation controls require DA3 scale perturbation to be disabled")

    out=(args.output_dir or resolve_project_path(cfg,cfg.experiment.output_dir)).resolve()
    results=(args.results_dir or resolve_project_path(cfg,cfg.experiment.results_dir)).resolve()
    if any((out/p).exists() for p in ("latest.pt","best.pt","training_history.csv","receipt.json")):
        raise FileExistsError(f"Refusing to overwrite training artifacts: {out}")
    out.mkdir(parents=True,exist_ok=True)
    results.mkdir(parents=True,exist_ok=True)
    epochs=int(args.epochs or cfg.train.epochs)
    train_ds=Dense4Dataset(cfg,"train")
    val_ds=Dense4Dataset(cfg,"val",augment=False)
    # Limit BOTH training examples and shuffle donors in smoke tests.
    if args.max_train_samples:
        train_ds.records=train_ds.records[:args.max_train_samples]
        train_ds.donor_indices={r["region"]:[i for i,s in enumerate(train_ds.records) if s["region"]!=r["region"]] for r in train_ds.records}
    if args.max_val_samples:
        val_ds.records=val_ds.records[:args.max_val_samples]
    if priorda_prior_frame:
        range_policy="D_prior min/range; direct DAv2-relative disparity reciprocal and de-normalization; retain all positive GT"
    elif priorda_identity:
        range_policy="D_prior min/range condition plus exact metric identity residual; retain all positive GT"
    elif priorda_zero_anchor:
        range_policy="BIM maximum scale with physical-zero origin; retain all positive GT"
    elif priorda_affine:
        range_policy="BIM-only shared affine frame; relative disparity output de-normalized to metric depth; retain all GT"
    else:
        range_policy="retain official metric sigmoid*20 head; do not discard any GT"
    range_audit=audit_depth_range(train_ds,policy=range_policy)
    print("TRAIN_GT_RANGE "+json.dumps(range_audit),flush=True)
    official_path,model_id,revision=resolve_checkpoint(cfg)
    model_cls=(
        PriorDARelativePriorFrameMetricRefiner if priorda_prior_frame
        else PriorDARelativePriorIdentityMetricRefiner if priorda_identity
        else PriorDARelativeZeroAnchorMetricRefiner if priorda_zero_anchor
        else PriorDARelativeMetricRefiner if priorda_affine
        else DAv2Dense4
    )
    model=model_cls.from_pretrained(model_id,revision=revision,local_files_only=True).to(device)
    init=model.initialization_audit(checkpoint_path=official_path,device=device)
    if not init["all_pass"]:
        raise RuntimeError(f"Official pretrained initialization failed: {init}")
    if cfg.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    generator=torch.Generator()
    workers=int(cfg.train.num_workers)
    train_loader=build_loader(train_ds,int(cfg.train.batch_size),workers,True,
        region_balanced=bool(cfg.train.region_balanced_sampling),
        region_balance_exponent=float(cfg.train.region_balance_exponent),
        samples_per_epoch=cfg.train.samples_per_epoch,generator=generator,persistent_workers=False)
    val_loader=build_loader(val_ds,int(cfg.train.val_batch_size),workers,False,
        generator=torch.Generator().manual_seed(seed+17),persistent_workers=False)
    if priorda_identity:
        model.eval()
        identity_batch=selected_batch(next(iter(val_loader)),device)
        with torch.inference_mode():
            identity_output=model(identity_batch)
        identity_difference=(identity_output["depth"]-identity_batch["base_depth"]).abs()
        identity_equal=torch.equal(identity_output["depth"],identity_batch["base_depth"])
        init["real_data_exact_prior_identity"]={
            "pass":bool(identity_equal),"bitwise_equal":bool(identity_equal),
            "frames":int(identity_batch["rgb"].shape[0]),
            "pixels":int(identity_batch["base_depth"].numel()),
            "max_abs_depth_diff_m":float(identity_difference.max()),
            "mean_abs_depth_diff_m":float(identity_difference.mean()),
        }
        init["all_pass"]=all(bool(value["pass"]) for key,value in init.items() if key!="all_pass")
        if not init["all_pass"]:
            raise RuntimeError(f"Real-data exact-prior initialization failed: {init}")
    if priorda_prior_frame:
        model.eval()
        semantic_batch=selected_batch(next(iter(val_loader)),device)
        with torch.inference_mode():
            semantic_output=model(semantic_batch)
        prior=semantic_batch["base_depth"]
        initial_abs_rel=(semantic_output["depth"]-prior).abs().div(prior).mean()
        init["real_data_direct_output_diagnostics"]={
            "pass":bool(torch.isfinite(semantic_output["depth"]).all()),
            "frames":int(prior.shape[0]),"pixels":int(prior.numel()),
            "output_vs_prior_abs_rel":float(initial_abs_rel),
            "output_disparity_zero_fraction":float((semantic_output["normalized_disparity"]<=0).float().mean()),
            "output_depth_min_m":float(semantic_output["depth"].min()),
            "output_depth_max_m":float(semantic_output["depth"].max()),
        }
        init["all_pass"]=all(bool(value["pass"]) for key,value in init.items() if key!="all_pass")
        if not init["all_pass"]:
            raise RuntimeError(f"Direct PriorDA initialization failed: {init}")
    groups=model.optimizer_parameter_groups(encoder_lr=float(cfg.train.encoder_learning_rate),
        decoder_lr=float(cfg.train.decoder_learning_rate),condition_lr=float(cfg.train.bim_condition_learning_rate))
    optimizer=torch.optim.AdamW(groups,weight_decay=float(cfg.train.weight_decay))
    accumulation=int(cfg.train.gradient_accumulation)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*math.ceil(len(train_loader)/accumulation))
    amp=bool(cfg.train.amp) and device.type=="cuda"
    amp_dtype_name=str(cfg.train.get("amp_dtype", "float16")).lower()
    amp_dtypes={"float16":torch.float16,"fp16":torch.float16,
                "bfloat16":torch.bfloat16,"bf16":torch.bfloat16}
    if amp_dtype_name not in amp_dtypes:
        raise ValueError(f"Unsupported train.amp_dtype: {amp_dtype_name}")
    amp_dtype=amp_dtypes[amp_dtype_name]
    # BF16 has FP32-like exponent range and does not need dynamic loss scaling.
    # This is important for the exact (unclamped) PriorDA reciprocal condition.
    scaler=torch.amp.GradScaler(device.type,enabled=amp and amp_dtype==torch.float16,
        init_scale=float(cfg.train.amp_initial_scale))
    runtime={"epochs":epochs,"micro_batch":int(cfg.train.batch_size),"accumulation":accumulation,
             "effective_batch":int(cfg.train.batch_size)*accumulation,"train_frames":len(train_ds),"val_frames":len(val_ds),
             "steps_per_epoch":math.ceil(len(train_loader)/accumulation),"train_batches":len(train_loader),
             "silog_reduction":"physical microbatch valid pixels, then gradient accumulation",
             "best_selection":"val frame_macro abs_rel", "seed":seed,"tf32":False,
             "deterministic_algorithms":False,"source_snapshot":False,
             "flash_sdp":torch.backends.cuda.flash_sdp_enabled(),
             "mem_efficient_sdp":torch.backends.cuda.mem_efficient_sdp_enabled(),
             "amp_dtype":str(amp_dtype).removeprefix("torch."),
             "gradient_scaler":bool(scaler.is_enabled()),
             "foundation":"DAv2-Relative ViT-B/14" if priorda_relative else "DAv2-Metric-Indoor ViT-B/14",
             "metric_frame":(
                 "per-frame D_prior min/range; direct official relative disparity inverse and de-normalization" if priorda_prior_frame
                 else "per-frame D_prior min/range condition; exact unrestricted metric log-depth identity residual" if priorda_identity
                 else "per-frame valid-BIM maximum with physical-zero origin" if priorda_zero_anchor
                 else "per-frame valid-BIM min/range" if priorda_affine
                 else "native metric head"
             )}
    materialized=plain(dict(cfg))
    materialized["runtime"]=runtime
    # Config is a small receipt, not a source snapshot. Eval uses the repo config.
    (out/"config.yaml").write_text(yaml.safe_dump(materialized,sort_keys=False),encoding="utf-8")
    atomic_json(out/"receipt.json",{"architecture":model.ARCHITECTURE,"initialization":init,
        "runtime":runtime,"depth_range_audit":range_audit,"split_provenance":train_ds.split_provenance})
    print("INITIALIZATION "+json.dumps(init),flush=True)
    print("RUNTIME "+json.dumps(runtime),flush=True)
    history=[]
    best_abs=float("inf")
    best_epoch=0
    steps=skipped=0
    aug=cfg.train.augment
    for epoch in range(1,epochs+1):
        epoch_seed=(seed+(epoch-1)*1_000_003)%2**32
        seed_everything(epoch_seed)
        generator.manual_seed(epoch_seed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started=time.perf_counter()
        samples=0
        sums={"silog":0.,"log_error_mean":0.,"log_error_var":0.}
        counts={k:0 for k in ("bim_shuffled","bim_full_dropped","bim_local_dropped","flipped","da3_perturbed")}
        q_sum=0.
        for index,raw_batch in enumerate(train_loader,1):
            batch=selected_batch(raw_batch,device)
            batch,log_q,applied=apply_da3_global_scale_perturbation(batch,
                probability=float(aug.da3_global_scale_probability),log_range=float(aug.da3_global_scale_log_range))
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp):
                output=model(batch)
                losses=dense_silog_loss(output["depth"],batch["gt_depth"],batch["gt_valid"],variance_focus=float(cfg.loss.variance_focus))
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError(f"Nonfinite SILog at epoch {epoch}, batch {index}")
            # Match historical accumulation including its partial final window /8.
            scaler.scale(losses["total"]/accumulation).backward()
            if index%accumulation==0 or index==len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.train.gradient_clip_norm))
                before=scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale()<before:
                    skipped+=1
                    print(f"AMP_SKIPPED epoch={epoch} batch={index} scale={scaler.get_scale()}",flush=True)
                else:
                    steps+=1
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            n=batch["rgb"].shape[0]
            samples+=n
            sums["silog"]+=float(losses["total"].detach())*n
            for k in ("log_error_mean","log_error_var"):
                sums[k]+=float(losses[k].detach())*n
            for k in counts:
                counts[k]+=int(applied.sum()) if k=="da3_perturbed" else int(raw_batch[k].sum())
            q_sum+=float(log_q.sum())
            if index==1 or index%int(cfg.train.log_every)==0 or index==len(train_loader):
                print(f"epoch={epoch}/{epochs} batch={index}/{len(train_loader)} silog={float(losses['total'].detach()):.6f} samples_per_s={samples/(time.perf_counter()-started):.2f} optimizer_steps={steps} skipped={skipped}",flush=True)
        train_seconds=time.perf_counter()-started
        val=evaluate(model,val_loader,device,amp,amp_dtype)
        score=float(val["final"]["frame_macro"]["abs_rel"])
        improved=score<best_abs
        if improved:
            best_abs,best_epoch=score,epoch
        row={"epoch":epoch,**{f"train_{k}":v/samples for k,v in sums.items()},
             "val_abs_rel":score,"val_pixel_micro_abs_rel":val["final"]["pixel_micro"]["abs_rel"],
             "val_rmse":val["final"]["frame_macro"]["rmse"],"val_delta1":val["final"]["frame_macro"]["delta1"],
             "train_seconds":train_seconds,"val_seconds":val["seconds"],"optimizer_steps":steps,"skipped_steps":skipped,
             "lr_encoder":optimizer.param_groups[0]["lr"],"train_samples":samples,"mean_log_q":q_sum/samples,**counts}
        history.append(row)
        atomic_json(out/f"val_epoch_{epoch}.json",val)
        atomic_json(out/"history.json",history)
        write_history(out/"training_history.csv",history)
        payload={"architecture":model.ARCHITECTURE,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                 "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"best_epoch":best_epoch,
                 "best_validation_abs_rel":best_abs,"config":materialized,"history":history,
                 "initialization":init,"optimizer_steps":steps,"skipped_steps":skipped,
                 "dav2_checkpoint_sha256":str(cfg.model.dav2.checkpoint_sha256)}
        atomic_torch_save(out/"latest.pt",payload)
        if improved:
            atomic_torch_save(out/"best.pt",payload)
        print("VAL "+json.dumps(row)+f" best_epoch={best_epoch}",flush=True)
    if not args.skip_test:
        test_ds=Dense4Dataset(cfg,"test",augment=False)
        test_loader=build_loader(test_ds,int(cfg.train.val_batch_size),workers,False,
            generator=torch.Generator().manual_seed(seed+31),persistent_workers=False)
        summary={"architecture":model.ARCHITECTURE,"best_epoch":best_epoch,"best_val_frame_macro_abs_rel":best_abs,
                 "history":history,"checkpoints":{}}
        for name in ("best","latest"):
            state=torch.load(out/f"{name}.pt",map_location="cpu",weights_only=False)
            model.load_state_dict(state["model"],strict=True)
            epoch=int(state["epoch"])
            del state
            metrics=evaluate(model,test_loader,device,amp,amp_dtype)
            summary["checkpoints"][name]={"epoch":epoch,"test":metrics}
            atomic_json(results/f"test_{name}.json",summary["checkpoints"][name])
            print(f"TEST checkpoint={name} epoch={epoch} "+json.dumps(metrics),flush=True)
        atomic_json(results/"summary.json",summary)
    print(f"TRAIN_COMPLETE best_epoch={best_epoch} best_val_frame_macro_abs_rel={best_abs} output={out}",flush=True)


if __name__=="__main__":
    main()
