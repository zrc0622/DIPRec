# 七组实验快速复现

所有命令都在仓库根目录执行。默认使用 SIDReasoner 官方划分、50 条历史、Qwen3-0.6B 和 seed 42；七组实验共用同一份 `.index.json`，不重新训练 SID 索引。

## 1. 安装环境

作用：创建独立的 Python 3.11 环境，先按远端 CUDA 版本安装匹配的 PyTorch 2.6.0，再安装其余依赖。不要把该 requirements 安装进 Open WebUI 等共享服务环境；其他应用造成的 `pip check` 冲突不代表独立 DIPRec 环境不兼容。

```bash
conda create -n diprec python=3.11 -y
conda activate diprec
# 先使用 PyTorch 官方命令安装与 CUDA 匹配的 torch==2.6.0 wheel。
python -m pip install --upgrade pip
python -m pip install -r requirements-diprec.txt
python -c "import torch, trl; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'TRL', trl.__version__, 'GPU', torch.cuda.is_available())"
```

`requirements-diprec.txt` 中的 `torch` 保持注释。四个 RL 实验统一使用固定版本 `trl==0.24.0` 与 `transformers==4.57.1`，不需要 VeRL、vLLM、FlashAttention、PEFT 或 W&B。DIPRec 在 TRL 上增加两阶段分层扩展，不再使用独立的手写优化循环。

这七组实验不安装 VeRL、vLLM、FlashAttention。若以后要运行仓库保留的原 SIDReasoner/VeRL 脚本，请再按 VeRL 官方文档单独构建对应环境。

这里的 MiniOneRec 是共享实验协议下的可比复现，不是逐行重跑上游脚本。启用的 SFT 任务族（历史 SID→SID、SID→title、title→SID、历史 SID→title）、RL 任务族（历史 SID→SID、title→SID、description→SID，以及最多 10,000 条 title-history→SID）、`G=16`、ranking reward 的结构和 catalog 约束均对齐官方实现。在 SID 奖励比较中，DIPRec 为兼容 tokenizer 插入的空格会忽略内部空白，而上游 MiniOneRec 将内部空白按字面比较。七种方法统一使用本仓库的 Qwen chat prompt、重建的长历史数据、AdamW 训练日程、checkpoint 协议和评测器。两个 RL baseline 各自冻结对应的父 checkpoint 作为 reference：Direct-RL 对应 Direct-SFT，MiniOneRec-RL 对应 MiniOneRec-SFT；这点有意不同于上游 MiniOneRec 的 `sync_ref_model=True` recipe。

RL 训练沿用 MiniOneRec 的 constrained beam sampling（`do_sample=True`）。七组评测的 SID 排序阶段统一使用确定性 constrained beam（`do_sample=False`）和 80 个原始 SID 候选预算；候选按 SID 去重后截取**最多** Top-10，不用重复项补满。DIPRec 将 80 个候选分配给各 plan（默认 `8 × 10`），再按 `log p(plan) + log p(SID | plan)` 联合排序；兴趣 plan 仍按固定 seed 采样。

默认显存配置偏保守：SFT 使用 micro-batch 2/累积 16，Direct/MiniOneRec-RL 使用 micro-batch 1/累积 16，DIPRec-RL 使用 micro-batch 1/累积 8。两个 RL trainer 都会自动令全局 generation batch 等于 `per_device_batch_size × world_size × gradient_accumulation_steps`（单卡分别为 16 和 8）。它们不是硬件最优值；先执行 `--dry_run`，再按远端 GPU 调整对应训练脚本参数。两个 DIPRec-RL 方法都使用冻结的 DIPRec-SFT reference（`beta=1e-3`），缓存 old-policy log-prob，并将每批 rollout 复用两次（`num_iterations=2`），因此第二次更新开始 PPO clipping 会实际生效。`diprec_traj_rl` 将完整轨迹 advantage 同时用于两阶段；`diprec_plan_rl` 保留 plan 跨 G、SID 在 B 内的两级 advantage。

## 2. 放置官方数据

作用：提供官方划分、固定 SID 和 MiniOneRec 的标题/描述对齐数据。

```text
data/Amazon/train/{dataset}_5_2016-10-2018-11.csv
data/Amazon/valid/{dataset}_5_2016-10-2018-11.csv
data/Amazon/test/{dataset}_5_2016-10-2018-11.csv
data/Amazon/index/{dataset}.index.json
data/Amazon/index/{dataset}.item.json
```

`{dataset}` 为 `Video_Games`、`Office_Products` 或 `Industrial_and_Scientific`。

## 3. 选择并构建长历史数据

作用：选出长历史最多的两个数据集，保持官方 train/valid/test target 不变，并将其前缀历史扩展到最多 50 条。

```bash
python scripts/select_long_history_datasets.py \
  --datasets Office_Products,Video_Games,Industrial_and_Scientific \
  --top_n 2 \
  --stats_output outputs/history_length_stats.csv \
  --selection_output configs/selected_long_history_datasets.txt

while read -r DATASET; do
  python scripts/build_long_history_data.py --dataset "$DATASET"
done < configs/selected_long_history_datasets.txt
```

输出：`data/processed/$DATASET/history_50/`。

## 4. 检查七组命令

作用：校验数据、任务数量、checkpoint 依赖和评测路径，不启动 GPU 训练。

```bash
bash scripts/run_all_comparisons.sh --dry_run
```

## 5. 运行七组实验

作用：依次运行以下依赖图。

| 方法 | 父 checkpoint | 训练目标 / 上游对齐边界 |
|---|---|---|
| `direct_sft` | Qwen | 历史 SID→下一 SID 的监督基线 |
| `direct_rl` | `direct_sft` | TRL GRPO，采用 MiniOneRec 风格 ranking reward 与 constrained beam sampling |
| `minionerec_sft` | Qwen | 共享协议下启用的四类 MiniOneRec SFT 任务 |
| `minionerec_rl` | `minionerec_sft` | 启用的四类 MiniOneRec RL 任务、`G=16`、冻结 reference |
| `diprec_sft` | `minionerec_sft` | 兴趣 plan SFT + plan 条件 SID SFT |
| `diprec_traj_rl` | `diprec_sft` | 轨迹级两阶段 TRL 目标 |
| `diprec_plan_rl` | `diprec_sft` | plan 级与 plan 内 SID 两级 advantage |

```text
Qwen
├─ direct_sft ───────────────→ direct_rl          (TRL)
└─ minionerec_sft ─┬────────→ minionerec_rl      (TRL)
                    └────────→ diprec_sft
                                  ├─ diprec_traj_rl
                                  └─ diprec_plan_rl
```

```bash
bash scripts/run_all_comparisons.sh
```

两个 DIPRec-RL 会分别从同一个 `diprec_sft` checkpoint 启动，不会串行继承彼此。输出位于：

```text
outputs/$DATASET/history_50/Qwen_Qwen3-0.6B/$METHOD/seed_42/
```

如需普通多卡 DDP，在相同命令前加启动参数。包装脚本会对 SFT 和 TRL-RL 训练调用 `accelerate launch`，预处理与评测仍保持单进程：

```bash
DIPREC_DDP=1 DIPREC_NUM_PROCESSES=4 bash scripts/run_all_comparisons.sh
```

Accelerate 配置必须使用普通 multi-GPU DDP。自定义集中式 rollout 只在 rank 0 生成候选后广播，因此每个 rank 都必须持有完整模型；trainer 初始化时会明确拒绝 DeepSpeed、FSDP 和 tensor parallelism。

非 dry-run 的依赖训练或评测在加载模型权重前，会校验 canonical method、processed-data 指纹（其中包含 SID index），以及需要时的 item-metadata 校验和；DIPRec 父 checkpoint 复用与评测还会校验兴趣标签策略/top-k/time-decay、conditioning 和 parameterization，RL 评测另校验训练时的 plan/beam 形状。任何不匹配都会立即失败，不会静默复用陈旧 checkpoint。结果 JSON 将不可变的 `training_config` 与本次 `evaluation_config` 分开保存。

## 6. 单独运行一个实验

作用：只运行目标方法；缺少的 SFT 父 checkpoint 会自动补训并校验。

```bash
bash scripts/run_experiment.sh \
  --method minionerec_sft \
  --dataset Video_Games

bash scripts/run_experiment.sh \
  --method diprec_plan_rl \
  --dataset Video_Games
```

可用方法：

```text
direct_sft
direct_rl
minionerec_sft
minionerec_rl
diprec_sft
diprec_traj_rl
diprec_plan_rl
```

## 7. 汇总结果

作用：将所有 `metrics.json` 汇总成一个 CSV。

```bash
python scripts/summarize_results.py \
  --input outputs/ \
  --output outputs/comparison.csv
```

## 8. 多随机种子和消融

作用：补跑多个 seed，或显式改用 leave-last-two-out。

```bash
bash scripts/run_all_comparisons.sh --seeds 42,43,44

bash scripts/run_all_comparisons.sh \
  --split_strategy leave_last_two_out \
  --seeds 42
```

常用参数：`--max_history_len 10|20|50`、`--model Qwen/Qwen3-1.7B`、`--conditioning history_visible|interest_bottleneck`。

RL 批参数包括 `--baseline_rl_per_device_batch_size`、`--baseline_rl_generation_batch_size`、`--baseline_rl_gradient_accumulation_steps`，以及对应的 `--diprec_rl_*` 参数。建议不传 generation batch，由程序安全推导；若显式指定，它必须包含完整的 `num_generations`/`num_plans` 分组，并等于全局有效 update batch。这样 TRL 会得到 `steps_per_generation = gradient_accumulation_steps`。其 sampler 内部使用 `repeat_count = num_iterations × steps_per_generation`：后一个因子用于依次提供所有 micro-step slice，前一个才表示同一 rollout 被多少次 optimizer update 复用。DIPRec 请保持 `num_iterations >= 2`，使同一 rollout 对应两次 optimizer update，第二次更新能实际触发 PPO clipping。

## 9. 运行检查

作用：运行轻量回归测试和语法检查。

```bash
python -m unittest discover -s tests -v
python -m compileall -q diprec scripts tests
```

固定 TRL 0.24.0 兼容环境下，当前发现的 79 项测试全部通过。另行执行的两个 RL trainer 完整双进程 CPU DDP 生命周期也均通过；Shell 语法、Python 编译、空白检查，以及 `Video_Games`/`Office_Products` × 七方法 dry-run 同样通过。CUDA kernel、GPU 显存上限和真实数据完整训练仍依赖远端训练机，需要在那里最终验证。
