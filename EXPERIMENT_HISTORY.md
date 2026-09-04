# DIPRec 实验历史台账

最后更新：2026-09-04

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

表中 `Val/Test` 均为生成评测结果；SFT 的 `best epoch` 是训练 loss 选择的
checkpoint，RL 没有 best-checkpoint 选择，评测的是最后一轮。

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

## 8. 已修复、待重新运行：四卡大 batch fixed-reference RL

这就是当前所称的“实验 3”。目标是保持其他条件不变，将每次 optimizer update
包含的完整 GRPO group 从单卡实验的 2 组增加到 32 组，以检验旧 RL 退化是否
主要来自有效奖励信号过少。

### 8.1 第一次启动：rank 0 rollout OOM

2026-09-04 首次使用下列配置启动；训练在第一个 rollout、进度 `0/3454` 时失败，
没有产生可计入结果表的 `metrics.json`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DIPREC_DDP=1 \
DIPREC_NUM_PROCESSES=4 \
bash scripts/run_experiment.sh \
  --method minionerec_rl \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag rl_fixed_ref_4gpu_eb512 \
  --baseline_rl_reference_mode fixed \
  --baseline_rl_per_device_batch_size 32 \
  --baseline_rl_gradient_accumulation_steps 4 \
  --baseline_rl_generation_batch_size 512
```

实际根因是 rank 0 在 constrained beam search 的模型前向中 CUDA OOM：GPU 0
共有 44.39 GiB，当时进程已使用 41.78 GiB，继续申请 1.50 GiB 失败。这不是
NCCL、batch 整除或 optimizer/backward 错误。旧 replicated-DDP 实现会收集
所有 rank 的 prompt，再只由 rank 0 执行生成；因此旧代码中 rank 0 一次承担
全局 512 个 candidate（32 个独立 prompt × 16 beams），其他三卡不执行生成。

### 8.2 代码修复后待运行：rank-local、micro-batch 32、accumulation 4

代码已将 Direct/MiniOneRec-RL 改为与官方 MiniOneRec 默认 non-vLLM 路径一致的
rank-local rollout：每个 rank 只生成自己的完整 GRPO group，之后仍由 TRL 跨 rank
汇总 reward 并归一化。恢复原来更快的 micro-batch 32、accumulation 4，并使用
新 run tag，避免与失败目录混合：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DIPREC_DDP=1 \
DIPREC_NUM_PROCESSES=4 \
bash scripts/run_experiment.sh \
  --method minionerec_rl \
  --dataset Office_Products \
  --sft_run_tag sft6e_lr1e-4_best \
  --run_tag rl_fixed_ref_4gpu_ranklocal_eb512_mb32_ga4 \
  --baseline_rl_reference_mode fixed \
  --baseline_rl_per_device_batch_size 32 \
  --baseline_rl_gradient_accumulation_steps 4 \
  --baseline_rl_generation_batch_size 512
```

修复前后的全局实验含义不变：

```text
4 GPUs × 32 candidates/GPU × 4 accumulation = 512 candidates/update
512 / G=16 = 32 complete GRPO prompt groups/update
旧代码：rank 0 生成 512 candidates，其他 rank 生成 0
新代码：每个 rank 生成 512 / 4 = 128 candidates（8 个完整 group）
```

曾考虑用 micro-batch 8、accumulation 16 保持全局 batch 512，但 TRL 会一次生成
整个 accumulation rollout，所以该配置每个 rank 仍生成 128 个 candidate，并不
降低生成峰值，反而增加 micro-step。若 rank-local 后每卡 128 个 candidate 仍 OOM，
应降低全局 generation/effective batch，而不是只交换 micro-batch 与累积步数。

预期新目录：

```text
outputs/Office_Products/history_50/Qwen_Qwen3-0.6B/minionerec_rl/seed_42_rl_fixed_ref_4gpu_ranklocal_eb512_mb32_ga4/
```

## 9. 证据完整性

- Office Products 的 7 个已完成 run 均有 `metrics.json`、`valid_metrics.json`
  和训练指标文件，命令参数由其中的 `training_config` 交叉验证。
- Video Games `1e-4` 完整结果可从 Git 历史中的 `outputs.zip` 读取；当前工作树
  中该 zip 已被删除，本文没有恢复它。
- Video Games `5e-5` 与 `2e-4` 只保留了对话中的现象和最终比较结论，缺少完整
  metrics，因此本文不填写无法验证的最终数值。
- 若用相同 run tag 在保留 checkpoint 的服务器上重新执行 SFT，runner 会拒绝
  覆盖；复跑时应给 `--run_tag` 增加新的后缀。
