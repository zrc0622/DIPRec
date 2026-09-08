# DIPRec 实验历史台账

最后更新：2026-09-08

本文只记录已经实际运行过的实验，以及当前已经确定但尚未运行的下一组实验。
训练产物中的 `training_config`、逐 epoch 日志和评测指标优先级最高；README
历史只用于恢复启动命令。结果 JSON 不保存物理 GPU 编号，因此无法确认时，下方
命令统一使用 GPU 0 作为等价复现设备，不代表原实验一定运行在 GPU 0。

## 1. 统一实验协议

- 模型：`Qwen/Qwen3-0.6B`
- 随机种子：42
- 数据划分：SIDReasoner official temporal split
- 最大历史长度：50
- SFT：单卡训练，cosine schedule，warmup ratio 0.03
- 最终生成评测：确定性 constrained beam，原始候选预算 80，最多返回 Top-10
- MiniOneRec-RL：`G=16`、temperature 1.0、learning rate `1e-5`、
  `beta=1e-3`、2 epochs

同一 run 的轻量日志和大模型权重分别位于：

```text
outputs/<dataset>/history_50/Qwen_Qwen3-0.6B/<method>/<run_id>/
output_dir/<dataset>/history_50/Qwen_Qwen3-0.6B/<method>/<run_id>/
```

## 2. 已完成实验总表

表中 `Val/Test` 均为生成评测结果；SFT 的 `best epoch` 是验证 loss 选择的
checkpoint，RL 没有 best-checkpoint 选择，评测的是训练停止时的 checkpoint。

| ID | 数据集 | 方法 / 变量 | best epoch | Val R@10 | Test R@10 | Test NDCG@10 |
|---|---|---|---:|---:|---:|---:|
| VG-SFT-1 | Video Games | MiniOneRec-SFT，LR `1e-4` | 3 | 0.11316 | 0.08808 | 0.05024 |
| OP-SFT-1 | Office Products | MiniOneRec-SFT，LR `1e-4` | 4 | 0.23592 | 0.17098 | 0.12972 |
| OP-DIP-S1 | Office Products | 旧 DIPRec-SFT，single plan | 3 | 0.15393 | 0.11015 | 0.06394 |
| OP-DIP-D1 | Office Products | 旧 DIPRec-SFT，diverse plan | 4 | 0.15968 | 0.11488 | 0.06808 |
| OP-DIP-P2 | Office Products | 新 paired activation，diverse | 2 | 0.23222 | **0.17222** | **0.13174** |
| OP-DIP-J2 | Office Products | 新 joint activation，diverse | 2 | 0.23448 | 0.17139 | 0.13035 |
| OP-RL-F1 | Office Products | MiniOneRec-RL，fixed reference | final | 0.22483 | 0.16728 | 0.12766 |
| OP-RL-S1 | Office Products | MiniOneRec-RL，periodic sync | final | 0.20304 | 0.14550 | 0.10432 |
| OP-RL-F4 | Office Products | MiniOneRec-RL，mixed-task fixed reference，四卡 | final | 0.22236 | 0.16584 | 0.12580 |
| OP-RL-H1 | Office Products | MiniOneRec-RL，history-only，四卡 | final | 0.22544 | 0.16482 | 0.12644 |
| OP-RL-C1 | Office Products | MiniOneRec-RL，保守 mixed，四卡一轮 | final | 0.23469 | 0.17078 | 0.12934 |
| OP-RL-PA | Office Products | 原奖励，保守 mixed，四卡 1,000 步 | final | 0.23448 | 未评测 | 未评测 |
| OP-RL-PB | Office Products | 主任务全未命中前缀辅助，λ=0.1，四卡 1,000 步 | final | 0.23469 | 未评测 | 未评测 |
| OP-RL-SW0 | Office Products | 串行强度验证：原奖励，1,000 步 | final | 0.23448 | 未评测 | 未评测 |
| OP-RL-SW3 | Office Products | 串行强度验证：λ=0.3，1,000 步 | final | 0.23407 | 未评测 | 未评测 |
| OP-RL-SW5 | Office Products | 串行强度验证：λ=0.5，1,000 步 | final | 0.23428 | 未评测 | 未评测 |
| OP-RL-SW10 | Office Products | 串行强度验证：λ=1.0，1,000 步 | final | 0.23490 | 未评测 | 未评测 |

### 当前结论

- 新的 paired/joint interest-activation SFT 恢复到了 MiniOneRec-SFT 水平；
  paired 与 joint 在 seed 42 下基本持平，不能据此宣称某个显著更好。
- 两种 activation SFT 都在 epoch 2 达到最低 `valid_sid_loss`，之后训练 loss
  继续下降而验证 SID loss 上升，因此应使用 epoch 2 的 `best_checkpoint`。
- 旧 single/diverse DIPRec-SFT 都明显弱于 MiniOneRec-SFT，且表现出更严重的
  热门候选集中。
- 两个单卡小 batch RL 都没有提升父 SFT。fixed-reference 轻微下降，sync-reference
  明显下降；旧 RL 每次更新只有 2 个完整 GRPO group，约 76%--78% 的验证 group
  reward 标准差为零，sync 还出现过很大的 KL 尖峰。
- 四卡 history-only RL 与四卡 mixed-task RL 基本持平，仍低于 SFT parent；去掉
  三类辅助任务没有消除负收益。保守 mixed 配置已完成，接近 SFT 但没有建立正向
  收益。固定这些条件的 1,000 步前缀辅助 A/B 已完成，仍未建立超过 SFT 或原奖励
  的收益：B 相对 A 在 4,866 条验证样本中净增 1 个 Top10 命中，但少 6 个 Top5
  命中。辅助信号实际启用，其优势绝对值总量约为原奖励项的 0.9%；详见第 12 节。
- 串行 λ=0.3/0.5/1.0 已完成，仍未建立推荐收益。λ=1.0 的 NDCG@10 为
  0.190032，比 SFT 仅高 0.000027，Top10/Top5 命中分别少 5/13 条，配对区间
  跨零。辅助强度约放大 10 倍但收益不明确，停止单纯扫描前缀系数；详见第 13 节。

## 3. Video Games：MiniOneRec-SFT 学习率实验

早期实际比较了 `5e-5`、`1e-4` 和 `2e-4`。当前只保留了 `1e-4` 的完整产物；
`5e-5` 与 `2e-4` 的原始 run tag 和结果目录没有保留下来。因此，下面三条命令
使用最终统一的 6-epoch 协议，是等价复现命令，不保证与早期两条 shell 命令逐字
一致。

```bash
# LR = 5e-5
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh \
  --method minionerec_sft \
  --dataset Video_Games \
  --run_tag sft6e_lr5e-5 \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 5e-5

# LR = 1e-4；完整结果已保留，最终采用
CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiment.sh \
  --method minionerec_sft \
  --dataset Video_Games \
  --run_tag sft6e_lr1e-4_best \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 1e-4

# LR = 2e-4
CUDA_VISIBLE_DEVICES=2 bash scripts/run_experiment.sh \
  --method minionerec_sft \
  --dataset Video_Games \
  --run_tag sft6e_lr2e-4 \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 2e-4
```

已保留的 `1e-4` 结果：

- best epoch：3，validation loss：1.90779
- validation：Recall@5 0.07913，Recall@10 0.11316，NDCG@10 0.06694
- test：Recall@5 0.05568，Recall@10 0.08808，NDCG@10 0.05024
- 用户记录的 `5e-5` 早期现象：epoch 1 train/valid loss 约 2.1/2.7，
  epoch 2 约 1.1/2.3
- 当时的最终比较结论：`1e-4` 的 validation/test NDCG 优于 `5e-5` 和
  `2e-4`

历史 `1e-4` 产物目前只存在于 Git 中已删除的 `outputs.zip` 归档，不应为了查看
它而恢复或覆盖当前 `outputs/`。

## 4. Office Products：MiniOneRec-SFT 基线

等价复现命令：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh \
  --method minionerec_sft \
  --dataset Office_Products \
  --run_tag sft6e_lr1e-4_best \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 1e-4
```

结果：

- best epoch：4，validation loss：1.68593
- validation：Recall@5 0.21023，Recall@10 0.23592，NDCG@10 0.19000
- test：Recall@5 0.14344，Recall@10 0.17098，NDCG@10 0.12972
- 该 `best_checkpoint` 是后续所有 Office Products DIPRec-SFT 和
  MiniOneRec-RL 的共同父模型

产物目录：

```text
outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/minionerec_sft/seed_42_sft6e_lr1e-4_best/
```

## 5. Office Products：旧版 DIPRec-SFT 消融

这两组是旧 `legacy` 目标：使用 `interest_bottleneck`，并把人工 primary plan
与未来 target SID 绑定。下方显式写出 `legacy`，避免今后 runner 默认值变化。

```bash
# 单一 plan；原 README 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh \
  --method diprec_sft \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag plan_single \
  --sft_objective legacy \
  --conditioning interest_bottleneck \
  --sft_plan_mode single \
  --sft_num_plans 8 \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 1e-4

# 多样 plan；原 README 使用 GPU 1
CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiment.sh \
  --method diprec_sft \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag plan_diverse \
  --sft_objective legacy \
  --conditioning interest_bottleneck \
  --sft_plan_mode diverse \
  --sft_num_plans 8 \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 1e-4
```

| run | best epoch / loss | Val R@10 | Test R@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| `seed_42_plan_single` | 3 / 0.69140 | 0.15393 | 0.11015 | 0.06394 |
| `seed_42_plan_diverse` | 4 / 0.45119 | 0.15968 | 0.11488 | 0.06808 |

这两个 run 的 validation loss 是旧混合任务 loss，不能与新 activation 的
`valid_sid_loss` 直接横向比较。

## 6. Office Products：新版兴趣激活 DIPRec-SFT

两组共享相同的 history-only 精简 plan 池和 diverse 轮换，只改变监督轨迹：

- paired：`history -> plan` 与 `history + plan -> target SID`
- joint：`history -> <think>plan</think>target SID`

```bash
# paired；原 README 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/run_experiment.sh \
  --method diprec_sft \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag interest_activation_plan_diverse \
  --sft_objective interest_activation \
  --conditioning history_visible \
  --sft_plan_mode diverse \
  --sft_num_plans 8 \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 8 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 1e-4

# joint；原 README 使用 GPU 1
CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiment.sh \
  --method diprec_sft \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag joint_interest_activation_plan_diverse \
  --sft_objective joint_interest_activation \
  --conditioning history_visible \
  --sft_plan_mode diverse \
  --sft_num_plans 8 \
  --sft_num_epochs 6 \
  --sft_micro_batch_size 4 \
  --sft_gradient_accumulation_steps 4 \
  --sft_learning_rate 1e-4
```

| run | selected metric | best epoch / loss | Val R@10 | Test R@10 | Test NDCG@10 |
|---|---|---:|---:|---:|---:|
| `seed_42_interest_activation_plan_diverse` | `valid_sid_loss` | 2 / 1.72885 | 0.23222 | 0.17222 | 0.13174 |
| `seed_42_joint_interest_activation_plan_diverse` | `valid_sid_loss` | 2 / 1.72144 | 0.23448 | 0.17139 | 0.13035 |

paired 最后一轮 train SID loss 已降到 0.55630，但 validation SID loss 回升到
1.88782；joint 对应为 0.61166 和 1.87683。两者都证明保存 best epoch 是必要的。

## 7. Office Products：MiniOneRec-RL reference 消融

两组都从 Office Products MiniOneRec-SFT 的 `best_checkpoint` 初始化。原 README
中 fixed-reference 使用 GPU 1，periodic-sync 使用 GPU 2。

```bash
# fixed reference
CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiment.sh \
  --method minionerec_rl \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag rl_fixed_ref \
  --baseline_rl_reference_mode fixed \
  --baseline_rl_per_device_batch_size 32 \
  --baseline_rl_gradient_accumulation_steps 1 \
  --baseline_rl_eval_steps 0.1

# 每 512 optimizer steps 同步 reference：ref <- 0.6*policy + 0.4*ref
CUDA_VISIBLE_DEVICES=2 bash scripts/run_experiment.sh \
  --method minionerec_rl \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag rl_sync_ref \
  --baseline_rl_reference_mode sync \
  --baseline_rl_ref_model_sync_steps 512 \
  --baseline_rl_ref_model_mixup_alpha 0.6 \
  --baseline_rl_per_device_batch_size 32 \
  --baseline_rl_gradient_accumulation_steps 1 \
  --baseline_rl_eval_steps 0.1
```

共同训练配置：单卡、effective/generation batch 32、每次更新 2 个完整 GRPO
group、2 epochs、55,290 optimizer steps。每组完成 10 次 validation eval，最终
评测 `final_checkpoint`，没有按 RL validation 选择 best checkpoint。

| run | Val R@10 | Test R@10 | Test NDCG@10 | 相对 SFT Test R@10 |
|---|---:|---:|---:|---:|
| `seed_42_rl_fixed_ref` | 0.22483 | 0.16728 | 0.12766 | -0.00370 |
| `seed_42_rl_sync_ref` | 0.20304 | 0.14550 | 0.10432 | -0.02548 |

## 8. 已完成：四卡大 batch、mixed-task fixed-reference RL

该实验将每次 optimizer update 包含的完整 GRPO group 从单卡实验的 2 组增加到
16 组，以检验旧 RL 退化是否主要来自有效 batch 太小。训练集仍沿用官方
MiniOneRec 的四任务 mixture。

Direct/MiniOneRec-RL 已采用与官方 MiniOneRec 默认 non-vLLM 路径一致的
rank-local rollout：每个 rank 只生成自己的完整 GRPO group，之后仍由 TRL 跨 rank
汇总 reward 并归一化。运行命令为：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DIPREC_DDP=1 \
DIPREC_NUM_PROCESSES=4 \
bash scripts/run_experiment.sh \
  --method minionerec_rl \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag rl_fixed_ref_4gpu_eb256_mb16_ga4 \
  --baseline_rl_reference_mode fixed \
  --baseline_rl_per_device_batch_size 16 \
  --baseline_rl_gradient_accumulation_steps 4 \
  --baseline_rl_generation_batch_size 256
```

配置含义：

```text
4 GPUs × 16 candidates/GPU × 4 accumulation = 256 candidates/update
256 / G=16 = 16 complete GRPO prompt groups/update
每个 rank 生成 256 / 4 = 64 candidates（4 个完整 group）
```

此前的 512-candidate rank-local 配置仍在最长本地 prompt batch 上 OOM，因此不再
作为推荐命令。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 用于降低显存
碎片，但实际峰值主要通过将每卡 rollout 从 128 降至 64 个 candidate 来降低。

结果目录：

```text
outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/minionerec_rl/seed_42_rl_fixed_ref_4gpu_eb256_mb16_ga4/
```

训练任务共 55,290 行：38,924 条 `history_sid_to_sid`、3,443 条
`title_to_sid`、2,923 条 `description_to_sid`、10,000 条
`title_history_to_sid`。结果仍低于 SFT parent：

| run | Val R@10 | Val NDCG@10 | Test R@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| MiniOneRec-SFT parent | 0.23592 | 0.19000 | 0.17098 | 0.12972 |
| 四卡 mixed-task RL | 0.22236 | 0.18418 | 0.16584 | 0.12580 |

增大 batch 没有修复负收益。周期日志显示约 76% 的 G=16 group 内 reward 方差为
0；同时 29.60% 的训练行属于最终推荐评测不包含的辅助任务。

## 9. 已完成：history-only MiniOneRec-RL

这是相对第 8 节的单变量消融。只把训练任务从 `official_mixed` 改成
`history_only`，保留 38,924 条 `SID history → next SID`；SFT parent、reward、
`G=16`、fixed reference、LR、2 epochs 和四卡 batch 与第 8 节完全相同。
validation/test 都保持原样。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DIPREC_DDP=1 \
DIPREC_NUM_PROCESSES=4 \
bash scripts/run_experiment.sh \
  --method minionerec_rl \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag rl_history_only_fixed_ref_4gpu_eb256_mb16_ga4 \
  --baseline_rl_task_scope history_only \
  --baseline_rl_reference_mode fixed \
  --baseline_rl_per_device_batch_size 16 \
  --baseline_rl_gradient_accumulation_steps 4 \
  --baseline_rl_generation_batch_size 256
```

结果目录：

```text
outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/minionerec_rl/seed_42_rl_history_only_fixed_ref_4gpu_eb256_mb16_ga4/
```

结果：

| run | Val R@10 | Val NDCG@10 | Test R@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| MiniOneRec-SFT parent | 0.23592 | 0.19000 | 0.17098 | 0.12972 |
| 四卡 mixed-task RL | 0.22236 | 0.18418 | 0.16584 | 0.12580 |
| 四卡 history-only RL | 0.22544 | 0.18481 | 0.16482 | 0.12644 |

history-only 相对 mixed 的 Valid R@10/NDCG@10 仅增加 `0.00308/0.00063`，Test
R@10 反而减少 `0.00103`，Test NDCG@10 仅增加 `0.00064`；这些差异不足以说明
history-only 更好，而且两种 RL 都低于 SFT parent。因此“辅助任务稀释”不是当前
负收益的主因，不再把 history-only 作为下一步推荐方向。该实验没有改变稀疏
reward，约 76% 的零方差 group 问题仍然存在。

## 10. 已完成：1-epoch 保守版 mixed-task MiniOneRec-RL

该组恢复官方四任务 `official_mixed`，继续使用第 8 节的 SFT parent、reward、
`G=16`、fixed reference 和四卡 effective batch 256，只改变三个优化参数：

```text
learning rate: 1e-5 -> 2e-6
KL beta:       1e-3 -> 1e-2
epochs:        2 -> 1
```

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DIPREC_DDP=1 \
DIPREC_NUM_PROCESSES=4 \
bash scripts/run_experiment.sh \
  --method minionerec_rl \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag rl_mixed_conservative_1e_lr2e-6_beta1e-2_4gpu_eb256 \
  --baseline_rl_task_scope official_mixed \
  --baseline_rl_reference_mode fixed \
  --baseline_rl_per_device_batch_size 16 \
  --baseline_rl_gradient_accumulation_steps 4 \
  --baseline_rl_generation_batch_size 256 \
  --baseline_rl_learning_rate 2e-6 \
  --baseline_rl_beta 1e-2 \
  --baseline_rl_num_epochs 1
```

预期目录：

```text
outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/minionerec_rl/seed_42_rl_mixed_conservative_1e_lr2e-6_beta1e-2_4gpu_eb256/
```

已复制并核验的最终结果：

| checkpoint | Valid R@10 | Valid NDCG@10 | Test R@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| SFT parent | 0.23592 | 0.19000 | 0.17098 | 0.12972 |
| 保守 mixed RL | 0.23469 | 0.18978 | 0.17078 | 0.12934 |

保守配置显著减少常规 RL 的退化，但没有建立超过 SFT 的收益；既有用户配对
bootstrap 区间跨零。本组未修改奖励，不能证明稀疏奖励是唯一失败原因。

2026-09-07 审计补充：前文约 76% 的零方差统计不能泛化为全部训练步。
找回完整单卡训练日志后，fixed/sync 全程零方差组均值为 61.76%/66.74%；
纯主推荐批次为 60.10%/65.07%。四卡完整逐步日志仍缺失。官方也共享稀疏奖励，
优化器/reference 精度等执行差异仍存在，尚未识别唯一或首要因果根源。

## 11. 证据完整性

- 早期 Office Products 运行具有 `metrics.json`、`valid_metrics.json`
  和训练指标文件；第 12 节 A/B 只评测 Valid，不填 Test 结果。命令参数由保存的
  `training_config` 交叉验证。
- Video Games `1e-4` 完整结果可从 Git 历史中的 `outputs.zip` 读取。本次用户提供
  了新的 `outputs.zip`，其中两组前缀 A/B 的 8 个文件与工作树对应文件逐字节一致。
- Video Games `5e-5` 与 `2e-4` 只保留了对话中的现象和最终比较结论，缺少完整
  metrics，因此本文不填写无法验证的最终数值。
- 若用相同 run tag 在保留 checkpoint 的服务器上重新执行 SFT，runner 会拒绝
  覆盖；复跑时应给 `--run_tag` 增加新的后缀。

## 12. 已完成：主推荐全未命中组的前缀辅助优势 A/B

用户授权修改实现、README 和实验历史。默认 `official` 不改变；显式选择
`main_miss_prefix` 才启用以下规则：

```text
h = 0.5 * I(first SID level matches) + 0.5 * I(first two SID levels match)
if task == history_sid_to_sid and no exact hit in the group:
    A = A_official + 0.1 * (h - mean(h))
else:
    A = A_official
```

原 all-miss 组的 `A_official=0`；辅助项不再按 std 归一化，避免强度系数抵消。
命中组、其他三类任务、KL、SFT、optimizer/reference 均保持原行为。
这是奖励/优势改进实验，不是声称官方执行已完全一致；分层 SID 奖励本身也已有
上游 GPR 扩展示例。

首个 A/B：两组相同保守超参数、同一现有 SFT、seed42、G16、四任务，均在
1,000 个 optimizer updates 停止，保留一轮 scheduler 总步数，只评测 Valid。
本轮采用固定停止步的 final checkpoint；尚未实现 250/500/1000 多 checkpoint
的自动 Recall/NDCG 选优，避免把周期 RL eval_loss 当成排名指标。
以下为已完成实验的复现命令；再次运行须使用新 run tag。

```bash
for reward_mode in official main_miss_prefix; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DIPREC_DDP=1 DIPREC_NUM_PROCESSES=4 \
  bash scripts/run_experiment.sh \
    --method minionerec_rl \
    --dataset Office_Products \
    --model Qwen/Qwen3-0.6B --max_history_len 50 --seed 42 \
    --sft_run_tag sft6e_lr1e-4_best \
    --run_tag "rl_prefix_ab_${reward_mode}_s1000_lr2e-6_beta1e-2_4gpu_eb256" \
    --baseline_rl_task_scope official_mixed \
    --baseline_rl_reference_mode fixed \
    --baseline_rl_per_device_batch_size 16 \
    --baseline_rl_gradient_accumulation_steps 4 \
    --baseline_rl_generation_batch_size 256 \
    --baseline_rl_learning_rate 2e-6 \
    --baseline_rl_beta 1e-2 \
    --baseline_rl_num_epochs 1 \
    --baseline_rl_reward_mode "$reward_mode" \
    --baseline_rl_prefix_reward_strength 0.1 \
    --baseline_rl_stop_after_steps 1000 \
    --baseline_rl_diagnostics \
    --eval_split valid || break
done
```

已完成实验的输出目录位于：

```text
outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/minionerec_rl/
  seed_42_rl_prefix_ab_official_s1000_lr2e-6_beta1e-2_4gpu_eb256/
  seed_42_rl_prefix_ab_main_miss_prefix_s1000_lr2e-6_beta1e-2_4gpu_eb256/
```

`rl_diagnostics.jsonl` 持久化 train/eval 日志和逐任务辅助信号统计；
原 `rl_training_metrics.json` 保持 eval-only。原 reward/zero-std 字段只描述
exact+rank，新增 `prefix_aux/.../aux_active` 才表示辅助信号是否实际启用。
训练 metadata 记录 reward mode、strength、stop_after_steps、完成步数和计划总步数。

**状态：两组均完成 1,000 次 optimizer update，λ=0.1 未建立推荐收益。**
实现阶段的 68 项 CPU 检查、双进程 TRL 生命周期验证和文档命令 dry-run 已通过。
2026-09-08 根据生产训练日志和保存的验证预测完成以下分析。

### 12.1 可比性和验证结果

两组 training_config 除 reward_mode 和三个输出路径外完全相同：同一 SFT parent
路径、数据指纹、seed42、四任务、G16、四卡有效 batch256、LR2e-6、beta0.01。
均完成 1,000/3,455 个计划更新，保留一轮 scheduler，约 0.2894 epoch；逐步任务组
数量、累计 token 数和学习率一致。验证集同为 4,866 条、1,795 个用户，生成预算
80→Top10。未取得生产权重哈希，因此不声称父模型权重逐位相同。

| checkpoint | Valid R@10 | Valid NDCG@10 | Valid R@5 | Valid NDCG@5 | Top10 命中数 |
|---|---:|---:|---:|---:|---:|
| SFT parent | 0.235923 | 0.190004 | 0.210234 | 0.181799 | 1,148 |
| A：原奖励 | 0.234484 | 0.189556 | 0.207974 | 0.181036 | 1,141 |
| B：前缀辅助 λ=0.1 | 0.234690 | 0.189771 | 0.206741 | 0.180743 | 1,142 |

B 相对 A 新增 12 个 Top10 命中、丢失 11 个，净增 1 个；Top5 净少 6 个。
B 相对 SFT 新增 13 个、丢失 19 个，净少 6 个。A/B Top10 候选集合平均重合
95.02%，主要推荐结果仍很接近。保守一轮 C1 恰好也有 1,142 个 Top10 命中，
但它与 B 是不同实验，不能据此认为两个 checkpoint 相同。

按用户配对 bootstrap 10,000 次，B−A 的 R@10 差为 +0.000206，95% 区间
[-0.001706, +0.002097]；NDCG@10 差为 +0.000215，区间
[-0.000514, +0.000957]。B−SFT 的两个区间也跨零。此次未建立正收益，亦不能
据此断言所有前缀系数/训练预算都无效；上述区间只反映验证用户抽样不确定性，
不包含训练种子间方差。

### 12.2 信号已启用，但推荐指标没有相应提升

- 两组各有 16,000 个训练 prompt 组。B 主推荐组 11,286 个，其中 exact 命中
  4,578 个（40.56%），另有 1,946 个全未命中组收到非零前缀优势（17.24%）。
  主任务有非零任务优势的组占比因此由 40.56% 增至 57.81%。其余三任务无辅助项。
- 按四任务实际组数加权，B 原奖励有信号的组占 37.01%，加入辅助后为 49.17%。
  这里统计的是任务优势，零任务优势组仍可能有 KL 梯度。原日志
  `frac_reward_zero_std=62.99%` 不包含新增辅助项，不能用它判断该功能未生效。
- 辅助项在激活组内平均 `|A_aux|=0.01304`；按当前 G16 exact+rank 公式计算，
  命中组平均 `|A_official|` 约 0.465–0.468。全训练候选辅助优势绝对值总量约为
  原奖励项的 0.917%–0.922%。这只衡量标量优势权重，不能解释成参数梯度占比。
- A/B 训练 KL 均值 0.002438/0.002460，最大值 0.03978/0.03724，这 1,000 步
  没有重现早期实验的大 KL 尖峰。两组各 999/1,000 步的裁剪前梯度范数大于 0.3；
  该统计不是实际参数更新量，不能单独证明裁剪导致无收益。

前轮离线 SFT Top10 的 23.59%→41.02% 是不同候选分布上的代理覆盖率，应与
以上实际训练 G16 统计分开。本轮覆盖增加没有转化为推荐收益，不能继续把
“奖励稀疏”当成已证实的唯一或首要原因。当前优先检验的假设是辅助信号强度
偏弱；另一可能是粗粒度前缀相似不足以指导具体商品的排序，两者尚未区分。

### 12.3 分析后的初始建议

不直接延长 λ=0.1 到整轮。先用现有参数开关将 λ 调至 0.5，仍从同一 SFT 开始，
固定 1,000 步、原 scheduler、seed42 和全部其他配置，只使用新 run tag；以
已完成的 A/λ=0.1 为对照。该幅度让相同候选上的辅助优势扩大 5 倍，目的是检验
强度假设，不保证收益；不同时修改 KL、学习率、任务范围或 SFT。

以 Valid NDCG@10 为主指标，并看 Recall@10、Top5 和配对命中得失。只有真实
推荐指标出现可信改善才扩预算；如果强度增大后只涨前缀指标，或仍无推荐收益，
则停止单纯提高前缀系数，转向研究奖励是否能区分同前缀下的具体商品。
目前没有启动新训练，也没有使用 test 调参。
随后按用户要求，将单点强度试验组织为第 13 节的串行四组验证，仍只改变奖励强度。

可复算脚本、原始文件 SHA256、详细证据和报告保存在本地忽略目录
`analysis/prefix_ab_results_20260908/`；本节保留可提交的核心结论。

## 13. 已完成：串行前缀强度验证

2026-09-08 用户要求脚本串行跑 3–4 组。新增
`scripts/run_prefix_sweep.py`，默认四组如下；已有 λ=0.1 与 SFT 指标作为历史参考，
此次重新运行原奖励作为本批对照。四组均已完成一次尝试，结果见 13.1 节。

| 顺序 | reward_mode | λ | optimizer updates | 目的 |
|---|---|---:|---:|---|
| 1 | official | 不生效 | 1,000 | 本批原奖励对照 |
| 2 | main_miss_prefix | 0.3 | 1,000 | 中等强度 |
| 3 | main_miss_prefix | 0.5 | 1,000 | 主要强度假设 |
| 4 | main_miss_prefix | 1.0 | 1,000 | 较强辅助的收益/退化边界 |

四组均从同一现有 `seed_42_sft6e_lr1e-4_best/best_checkpoint` 独立初始化。
固定 Office Products、Qwen3-0.6B、history50、seed42、四任务、G16、四卡
micro16×GA4、有效 batch256、LR2e-6、beta0.01、fixed reference 和一轮 scheduler。
只做 Valid 预算80→Top10。原奖励组记录 λ=0.1，但该模式不使用辅助项。

以下命令保留用于复现；新训练应改用新 sweep tag。

```bash
python3 scripts/run_prefix_sweep.py --sweep_tag prefix_strength_v1 --gpus 0,1,2,3
```

加 `--dry_run` 只打印命令；加 `--resume` 跳过校验通过的已完成组，并将失败/
中断组放入新 attempt 目录，从父 SFT 重跑。只要三组则加 `--strengths 0.5 1.0`，
恢复时保持相同列表。新实验使用新 sweep tag；不指定则自动生成时间戳。
`--summarize_only` 仅重建汇总，不需要训练权重。

执行中每组训练和最终验证结束后才启动下一组，失败即停止。新增通用 runner
开关 `--require_existing_sft`，在父模型缺失或不兼容时明确退出；本脚本不启动
SFT 训练。相同 sweep tag 有进程锁，旧输出不覆盖；恢复不等同于恢复未完成组的
optimizer state，未完成组从 SFT 重新训练，之前 attempt 保留。

汇总目录为
`outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/prefix_sweeps/<sweep_tag>/`：

- `state.json`：每组状态、命令、run tag 和尝试记录；逐组控制台输出另存 `.log`。
- `summary.csv` / `summary.json`：Valid R/NDCG@5/10、SID 前缀命中、相对 SFT/
  本批原奖励的 Top10 指标差，以及按任务组数加权的辅助覆盖、幅度和 KL。
- 原训练日志、预测、metrics 和权重仍使用标准 `minionerec_rl` 目录；历史参考
  只在配置匹配时纳入，缺失时注明，不用历史结果冒充新试验。

这组实验检验强度，而非保证更大的 λ 更好。先看 Valid NDCG@10，再看 Recall@10、
Top5 与配对得失；自动汇总不计算显著性或自动扩大预算。若增强后仍只有前缀
指标改善，应停止单纯增大系数。实现阶段的 35 项串行调度/runner 检查通过，
现已取得四组生产训练与最终验证结果。

### 13.1 验证结果与可比性

| 模型 | Valid R@10 | Valid NDCG@10 | Valid R@5 | Valid NDCG@5 | Top10 命中数 |
|---|---:|---:|---:|---:|---:|
| SFT parent | 0.235923 | 0.190004 | 0.210234 | 0.181799 | 1,148 |
| 本批原奖励 | 0.234484 | 0.189556 | 0.207974 | 0.181036 | 1,141 |
| λ=0.1（上轮） | 0.234690 | 0.189771 | 0.206741 | 0.180743 | 1,142 |
| λ=0.3 | 0.234073 | 0.189279 | 0.207152 | 0.180623 | 1,139 |
| λ=0.5 | 0.234279 | 0.189619 | 0.207974 | 0.181211 | 1,140 |
| λ=1.0 | 0.234895 | 0.190032 | 0.207563 | 0.181305 | 1,143 |

四组 training_config 除奖励模式/强度和输出路径外完全一致；均完成
1,000/3,455 步，逐步任务组数、累计 token 数、学习率相同。验证集仍为
4,866 条、1,795 用户、预算80→Top10。此次原奖励的 1,000 条训练记录与旧原奖励
完全一致，验证预测文件也逐字节相同；因此不是新增独立随机种子样本，亦没有
证据把本批无收益归于重跑波动。未取得生产权重哈希，不声称权重逐位相同。

λ=1.0 相对本批原奖励新增13、丢失11个Top10命中，净增2；相对SFT新增13、
丢失18个，净少5。其Top1比SFT多9条、Top5少13条，是不同排名位置的变化，
不能因NDCG@10数值略高就认定总体改善。所有RL的R@5/NDCG@5仍低于SFT。

按用户配对bootstrap 10,000次：λ=1.0−原奖励的R@10差为+0.000411，95%区间
[-0.001616,+0.002437]；NDCG@10差为+0.000476，区间[-0.000291,+0.001296]。
λ=1.0−SFT的NDCG@10差仅+0.00002745，区间[-0.000906,+0.000972]。
所有新强度对原奖励/SFT的Top10指标区间均跨零；这是逐比较区间，未做多重比较
校正，不含训练随机性。不能将多档中最高点估计直接当作可靠最优方案。

### 13.2 强度已增加，仍未观察到前缀到商品命中的转化收益

| 模式 | 主任务辅助激活组率 | 全任务非零任务优势组率 | 激活组平均绝对辅助优势 | 辅助/原奖励绝对优势总量比 | KL均值/最大值 |
|---|---:|---:|---:|---:|---|
| 原奖励 | 0% | 36.91% | — | 0% | 0.002438 / 0.03978 |
| λ=0.1（上轮） | 17.24% | 49.17% | 0.01304 | 约0.92% | 0.002460 / 0.03724 |
| λ=0.3 | 17.21% | 49.14% | 0.03894 | 约2.74% | 0.002418 / 0.03182 |
| λ=0.5 | 17.46% | 49.32% | 0.06448 | 约4.60% | 0.002380 / 0.02874 |
| λ=1.0 | 17.23% | 49.02% | 0.13007 | 约9.20% | 0.002379 / 0.02646 |

按各任务实际组数加权；“非零”指任务优势，原全零组仍可能有KL梯度。强度确实
扩大约10倍，未被std归一化抵消；9.2%是标量绝对优势总量比，不是梯度贡献。
主推荐训练exact命中组率从原奖励40.50%到λ=1.0的40.40%，平均前缀分数
0.13418→0.13368，也未观察到明显上升。全程rollout均值不是冻结探针学习曲线，
不能据此断言完全没有学习，但当前没有增强信号带来覆盖提升的证据。

固定训练前SFT验证Top10分组：未exact命中但已有正确前缀的857条样本中，
原奖励最终命中15条，λ=0.1/0.3/0.5/1.0分别命中11/9/12/10条。λ=1.0并未
更好地找回这类接近答案的样本。这里是验证集事后分层，不是训练辅助激活组的
因果跟踪；随机G16和确定性Top10不同，前缀存在也不等于组内一定有分数差。

这轮没有大KL尖峰，当前亦不支持把结果归因于KL失控。更重要的是，覆盖与强度
都已得到验证，但没有产生明确推荐收益，“只是λ=0.1太小”的解释缺乏支持。
仍不能严格区分前缀奖励方向价值有限与当前更新预算/优化动态的限制。

### 13.3 后续建议

停止单纯扫描前缀系数，保留SFT基线，不因λ=1.0多出0.000027 NDCG就替换基线
或直接扩至整轮。当前优先研究同前缀商品之间的奖励区分能力，例如冻结商品
内容向量与目标商品相似度：先离线检查它在同前缀候选中是否确有分辨力、是否
被热门或泛化描述主导，再决定固定预算A/B。保留exact+rank、主任务all-miss
门控和当前SFT/训练配置。该方向尚未实现或验证有效，本次没有修改训练代码。

新zip的24个本批相关文件与工作树一致；重算脚本、SHA256、配对统计和详细报告
位于本地忽略目录`analysis/prefix_strength_results_20260908/`，不使用test调参。


## 14. 五组 reference / KL / 学习率优化（已实现，待运行）

用户明确不做向量相似度实验，因此本轮取代13.3节的后续方向。脚本
`scripts/run_rl_optimization_sweep.py`已实现，尚无这五组生产训练结果。

| 组 | Reference | β | LR | 单因素比较 |
|---|---|---:|---:|---|
| A | fixed | .01 | 2e-6 | 对照 |
| B | sync512，α=.6 | .01 | 2e-6 | B−A |
| C | fixed | .001 | 2e-6 | C−A |
| D | sync512，α=.6 | .001 | 2e-6 | D−C / D−B |
| E | fixed | .01 | 5e-6 | E−A |

保持当前Office Products、Qwen3-0.6B、history50、seed42和已有MiniOneRec-SFT；
统一四任务、原exact+rank、G16、四GPU、micro16×GA4、全局batch256，1 epoch
共3,455更新，原cosine/warmup和训练中RL validation日程保留。无prefix shaping，
无向量相似度奖励。所有组从同一SFT独立初始化，sync为`.4ref+.6policy`。

新功能只增加可选模型快照和离线诊断；未改SFT训练或RL奖励公式。1,000/2,000步
保存模型与tokenizer，整轮结束后评测两个快照和final；统一Valid80候选Top10，
主终点3,455，阶段结果按相同步数比较。启动前验证总步数，防止数据改变后预算漂移。

冻结train/valid各128条主任务样本；初始SFT通过相同推荐prompt的beam80生成
5个不同错误候选，加真实目标并固定。每阶段测全词表SID+EOS logp、target概率、
对最强错误候选的margin、固定候选rank及相对SFT变化。六候选归一化KL只表示
固定候选子集的分布变化，不是全策略KL；不能用sync训练KL替代对初始SFT的比较。

```bash
python3 scripts/run_rl_optimization_sweep.py --sweep_tag rl_opt_v1 --dry_run
python3 scripts/run_rl_optimization_sweep.py --sweep_tag rl_opt_v1 --gpus 0,1,2,3
python3 scripts/run_rl_optimization_sweep.py --sweep_tag rl_opt_v1 --gpus 0,1,2,3 --resume
python3 scripts/run_rl_optimization_sweep.py --sweep_tag rl_opt_v1 --summarize_only
```

已完成训练但评测失败时只补评测；训练中断使用新attempt从SFT开始，旧产物保留。
快照没有optimizer/reference状态，不支持训练中途续接。summary在
`outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/rl_optimization_sweeps/rl_opt_v1/`，
包含15阶段行、SFT差值、B−A/C−A/D−C/D−B/E−A比较、三区间各任务命中组率和KL。
已保存预测供后续配对分析，汇总本身不是显著性检验，不自动选赢家，也不评测test。

本地验证：53项相关回归检查通过，覆盖真实tiny-Qwen的SID+EOS分数、完整探针构建/评测、
快照重载、随机数与模型更新一致性、失败续跑、旧runner与TRL生命周期；另通过双进程
CPU DDP快照/同步检查。shell语法、Python AST和空白检查通过。正式四GPU训练未运行，
因此本节没有新的推荐效果结论。检查日志保存在本地忽略目录
`analysis/rl_optimization_implementation_20260908/`。
