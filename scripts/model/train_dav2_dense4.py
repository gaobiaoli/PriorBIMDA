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
def evaluate(model, loader, device, amp):
    model.eval()
    final, raw = BothMetrics(), BothMetrics()
    start = time.perf_counter()
    for raw_batch in loader:
        batch = selected_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = model(batch)["depth"]
        support = batch["gt_valid"].bool() & torch.isfinite(batch["gt_depth"]) & (batch["gt_depth"] > 0)
        final.update(pred, batch["gt_depth"], support)
        raw.update(batch["base_depth"], batch["gt_depth"], support)
    return {"final": final.compute(), "raw_da3": raw.compute(), "seconds": time.perf_counter()-start,
            "support": "official all finite positive GT, no depth cutoff", "alignment": "none"}


def audit_depth_range(dataset):
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
            "policy":"retain official metric sigmoid*20 head; do not discard any GT"}


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
    range_audit=audit_depth_range(train_ds)
    print("TRAIN_GT_RANGE "+json.dumps(range_audit),flush=True)
    official_path,model_id,revision=resolve_checkpoint(cfg)
    model=DAv2Dense4.from_pretrained(model_id,revision=revision,local_files_only=True).to(device)
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
    groups=model.optimizer_parameter_groups(encoder_lr=float(cfg.train.encoder_learning_rate),
        decoder_lr=float(cfg.train.decoder_learning_rate),condition_lr=float(cfg.train.bim_condition_learning_rate))
    optimizer=torch.optim.AdamW(groups,weight_decay=float(cfg.train.weight_decay))
    accumulation=int(cfg.train.gradient_accumulation)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*math.ceil(len(train_loader)/accumulation))
    amp=bool(cfg.train.amp) and device.type=="cuda"
    scaler=torch.amp.GradScaler(device.type,enabled=amp,init_scale=float(cfg.train.amp_initial_scale))
    runtime={"epochs":epochs,"micro_batch":int(cfg.train.batch_size),"accumulation":accumulation,
             "effective_batch":int(cfg.train.batch_size)*accumulation,"train_frames":len(train_ds),"val_frames":len(val_ds),
             "steps_per_epoch":math.ceil(len(train_loader)/accumulation),"train_batches":len(train_loader),
             "silog_reduction":"physical microbatch valid pixels, then gradient accumulation",
             "best_selection":"val frame_macro abs_rel", "seed":seed,"tf32":False,
             "deterministic_algorithms":False,"source_snapshot":False,
             "flash_sdp":torch.backends.cuda.flash_sdp_enabled(),
             "mem_efficient_sdp":torch.backends.cuda.mem_efficient_sdp_enabled()}
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
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
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
        val=evaluate(model,val_loader,device,amp)
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
            metrics=evaluate(model,test_loader,device,amp)
            summary["checkpoints"][name]={"epoch":epoch,"test":metrics}
            atomic_json(results/f"test_{name}.json",summary["checkpoints"][name])
            print(f"TEST checkpoint={name} epoch={epoch} "+json.dumps(metrics),flush=True)
        atomic_json(results/"summary.json",summary)
    print(f"TRAIN_COMPLETE best_epoch={best_epoch} best_val_frame_macro_abs_rel={best_abs} output={out}",flush=True)


if __name__=="__main__":
    main()
