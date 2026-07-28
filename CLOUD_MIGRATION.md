# 云服务器迁移指南

## 推荐云端规格

- Ubuntu 22.04 或兼容 Linux；
- NVIDIA GPU 12 GB 起步；若接入 DA3 特征建议 24 GB；
- 系统内存至少 32 GB，推荐 64 GB；
- 仅继续 V3：磁盘至少 30 GB；
- 保留原始 PCD/BIM 并重新制备 GT：建议 100 GB 以上；
- 数据盘与代码盘分离，训练输出定期同步到对象存储。

## 建议目录

```text
/workspace/
├── BIM-PriorDA3/
├── depth-anything-3/
└── SLABIM/
    ├── BIM/
    ├── calibration_files/
    └── sensor_data/
```

设置：

```bash
export BIM_PRIORDA3_SLABIM_ROOT=/workspace/SLABIM
```

项目已支持该环境变量，并会在 manifest 的旧绝对路径失效时，自动将样本重定位到当前
`data/processed/.../samples/`，将 RGB 重定位到上述 SLABIM 目录。

## 需要复制什么

### A. 继续当前 V3 训练/评测

必需：

```text
BIM-PriorDA3/                                  约 3.8 GB
SLABIM/sensor_data/*/images/                   约 4.3 GB
```

建议同时复制：

```text
SLABIM/calibration_files/                      12 KB
SLABIM/BIM/                                    约 477 MB
depth-anything-3/                              约 48 MB
```

`slabim_504_r50` 的 NPZ 已缓存 DA3、BIM、强锚点、V1 候选和 ±50 PCD GT，因此继续 V3
训练不需要重新运行 DA3，也不需要 PCD。

### B. 重新生成 ±50 PCD GT

另需：

```text
SLABIM/sensor_data/*/points/                   约 12.8 GB
```

其中应包含：

- `data/*.pcd`
- `timestamps.txt`
- `lidar_pose_local_to_bim_from_rosbag.txt`
- `lidar_pose_local_to_slam_smoothed.txt`

不需要复制 ROS bag。

### C. 对新图像端到端推理

还需可用的 DA3 安装和模型权重。Hugging Face/模型缓存未包含在48 MB代码仓库中，可在云端
重新下载，或单独迁移本机模型缓存。

## 推荐传输方式

工作站到云端可使用：

```bash
rsync -aH --info=progress2 --partial \
  /home/bgao491/BIM-PriorDA3/ \
  USER@HOST:/workspace/BIM-PriorDA3/

rsync -aH --info=progress2 --partial \
  --include='*/' --include='*.png' --include='timestamps.txt' --exclude='*' \
  /home/bgao491/SLABIM/sensor_data/ \
  USER@HOST:/workspace/SLABIM/sensor_data/

rsync -aH --info=progress2 --partial \
  /home/bgao491/SLABIM/BIM/ \
  USER@HOST:/workspace/SLABIM/BIM/

rsync -aH --info=progress2 --partial \
  /home/bgao491/SLABIM/calibration_files/ \
  USER@HOST:/workspace/SLABIM/calibration_files/
```

如果云平台只支持对象存储，先用 `tar --zstd` 打包可减少大量小文件传输开销，但压缩包应在
校验和确认后再删除。

## 环境安装

建议先按云端驱动安装匹配的 PyTorch，再安装项目依赖：

```bash
cd /workspace/BIM-PriorDA3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 按云端CUDA/驱动安装PyTorch，不要机械复制本机CUDA wheel。
python -m pip install numpy==1.26.4 opencv-python-headless \
  scipy==1.13.1 PyYAML==6.0.3 open3d==0.19.0 plyfile pytest
python -m pip install -e '.[slabim]'
python -m pip install -e /workspace/depth-anything-3
```

若只训练已有 V3 缓存，不重新运行 DA3，可以暂不安装 depth-anything-3。
若位姿文件已存在且不重新解析 rosbag，也可只安装 `pip install -e .`。

## 源端校验和

传输前后分别运行：

```bash
cd /workspace/BIM-PriorDA3
sha256sum \
  configs/slabim_single_frame_r50_v3.yaml \
  data/processed/slabim_504_r50/manifest.jsonl \
  outputs/slabim_single_frame_r50/best.pt \
  outputs/slabim_single_frame_r50_v3/best.pt \
  outputs/slabim_single_frame_r50_v3/evaluation/summary.json \
  outputs/slabim_single_frame_r50_v3/evaluation_safe/summary.json
```

两端输出必须完全一致。需要校验全部NPZ时：

```bash
find data/processed/slabim_504_r50/samples -type f -name '*.npz' -print0 \
  | sort -z | xargs -0 sha256sum > all_samples.sha256
sha256sum -c all_samples.sha256
```

## 云端验收

```bash
cd /workspace/BIM-PriorDA3
export BIM_PRIORDA3_SLABIM_ROOT=/workspace/SLABIM
source .venv/bin/activate

python scripts/verify_cloud_setup.py

python scripts/run_slabim_experiments.py \
  --slabim-root /workspace/SLABIM \
  --stages verify

python scripts/evaluate.py \
  --config configs/slabim_single_frame_r50_v3.yaml \
  --checkpoint outputs/slabim_single_frame_r50_v3/best.pt \
  --split val \
  --output outputs/cloud_validation_check
```

验证集安全模型 AbsRel 应约为 0.08928。若差异明显，先检查路径、PyTorch版本、缓存文件和
checkpoint校验和，不要立即重新训练。

## 断点和备份

- 每个实验使用新的 `experiment.name/output_dir`，不要覆盖现有 V1/V3；
- 至少保留 `best.pt`、`last.pt`、配置、history和evaluation；
- 每个epoch或每小时将小型checkpoint/JSON同步至持久存储；
- 云实例关机前确认数据盘是否为持久盘；
- 不要把Hugging Face token、云密钥或私有URL写入配置/交接文件。
