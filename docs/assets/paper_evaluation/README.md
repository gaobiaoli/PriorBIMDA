# Paper/PPT assets

本目录由 `scripts/analysis/generate_paper_assets.py` 从当前统一协议结果生成。每张图同时提供
PNG、SVG 和 PDF；SVG/PDF 适合后续在 PPT 或论文中重新排版。

- `process/candidate_a_method_pipeline.*`：简洁方法流水线；
- `process/candidate_b_dual_stream_architecture.*`：输入/网络结构细节；
- `process/candidate_c_evaluation_protocol.*`：无泄漏科研评测流程；
- `quantitative/main_blind_test_absrel.*`：两域主结果；
- `quantitative/area1_subset_absrel.*`：家具/冲突子集；
- `quantitative/area1_room_pairs_and_bootstrap.*`：房间配对与区间；
- `quantitative/deterministic_bim_direct_factor_ablation.*`：非学习 BIM-direct 的逐因素
  validation 消融（post-hoc）；
- `quantitative/area1_train_only_scale_{heatmap,pareto}.*`：train-only 尺度敏感性；
- `quantitative/registered_training_curves.*`：两域单次训练验证曲线。

`manifest.json` 记录输入文件 SHA、图中数值和风险提示。三张 process 图是互斥备选素材，不是
需要同时使用的拼版。若结果文件变化，应重新运行生成器，不能手工沿用旧图。
