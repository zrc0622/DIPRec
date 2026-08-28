# DIPRec 远程复现指南

本扩展不会改动 SIDReasoner 原有的训练文件。以下命令均在仓库根目录执行。训练环境建议使用 Linux、Python 3.10 和 CUDA GPU，并确保 Qwen3、VeRL、vLLM 与当前 CUDA/PyTorch 版本兼容。

## 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-diprec.txt
```

如需运行多卡 SIDReasoner GRPO，请参考原始 README 使用兼容 VeRL 的 CUDA 镜像，并设置 `NUM_GPUS`。数据需自行放到本地；Hugging Face 会优先读取本地缓存，缺少模型时则在远程主机下载。

仓库内置的 VeRL 代码主要适配 PyTorch 2.4–2.6 和 vLLM 0.8。若镜像未预装 FlashAttention/FlashInfer，请安装与 CUDA 环境匹配的版本后再运行 RL。

## 数据准备与长历史数据集选择

将未经历史截断的 Amazon 事件级交互文件放到：

```text
data/Amazon/raw/Office_Products.jsonl[.gz]
data/Amazon/raw/Video_Games.jsonl[.gz]
data/Amazon/raw/Industrial_and_Scientific.jsonl[.gz]
```

每条记录至少需要包含：

- 用户字段：`user_id` 或 `reviewerID`
- 商品字段：`item_id` 或 `asin`
- 时间字段：`timestamp` 或 `unixReviewTime`

将已有 SID 映射放到 `data/Amazon/index/{dataset}.index.json`。带有 `history_item_*` 字段的旧 CSV 已经预先构造或截断历史，不能作为原始事件文件，因此程序会拒绝读取。

SIDReasoner 的论文和原脚本有时将 `Video_Games` 简写成 `Games`。新脚本同时接受这两个名称，并统一规范为 `Video_Games`；这里的 `Games` 并不是 `Toys_and_Games`。

先统计三个数据集的真实历史长度并选择长历史最丰富的一至两个数据集：

```bash
python scripts/select_long_history_datasets.py \
  --datasets Office_Products,Video_Games,Industrial_and_Scientific \
  --top_n 2 \
  --stats_output outputs/history_length_stats.csv \
  --selection_output configs/selected_long_history_datasets.txt

DATASET="$(sed -n '1p' configs/selected_long_history_datasets.txt)"

python scripts/build_long_history_data.py \
  --dataset "$DATASET" \
  --max_history_len 50
```

数据构建流程按用户和时间排序，采用 leave-last-two-out：倒数第二个交互作为验证目标，最后一个作为测试目标，每个目标只能看到其之前的历史，再保留最近 50 个行为。输出位于 `data/processed/$DATASET/history_50/`，包括：

- `train.jsonl`、`valid.jsonl`、`test.jsonl`
- 带原始数据和 SID 映射校验和的 `manifest.json`
- 截断前后历史长度统计 `history_length_stats.csv`

`configs/` 中提供 10、20、50 三种历史长度配置。修改长度后需要重新构建对应数据。

重复运行时，如果已有 `manifest.json`，预处理会复用并校验现有数据。共享 checkpoint 只有在模型权重、`config.json` 和 `training_config.json` 均存在时才会复用；若目录不完整，程序会提前报错。需要重建或重训时，请先手动删除或移动对应的旧目录。

## 单数据集对比实验（seed 42）

各方法使用同一份数据、同一个 catalog trie、相同的 80 条原始 SID 候选搜索预算，并统一输出 Top-10。DIPRec 会把总预算分配到不同兴趣 plan，再对所有轨迹统一重排，避免比基线多获得 `num_plans` 倍的候选探索量。

结果默认写入：

```text
outputs/$DATASET/history_50/Qwen_Qwen3-0.6B/$METHOD/seed_42/
```

可通过 `--eval_candidate_budget` 修改公共搜索预算；对于 DIPRec，该值不能小于 `--num_plans`。

```bash
# 1. Direct-SID 基线
bash scripts/run_experiment.sh --method direct_sid --model Qwen/Qwen3-0.6B --dataset "$DATASET" --max_history_len 50 --max_seq_len 2048 --seed 42

# 2. SIDReasoner：共享数据上的自然语言 reasoning SFT + 原始 VeRL GRPO
bash scripts/run_experiment.sh --method sidreasoner --model Qwen/Qwen3-0.6B --dataset "$DATASET" --max_history_len 50 --max_seq_len 2048 --seed 42

# 3. DIPRec SFT
bash scripts/run_experiment.sh --method diprec_sft --model Qwen/Qwen3-0.6B --dataset "$DATASET" --interest_topk 3 --max_history_len 50 --max_seq_len 2048 --seed 42

# 4. DIPRec trajectory-level GRPO
bash scripts/run_experiment.sh --method diprec_trajectory_grpo --model Qwen/Qwen3-0.6B --dataset "$DATASET" --interest_topk 3 --num_plans 8 --sid_beams 8 --max_history_len 50 --max_seq_len 2048 --seed 42

# 5. DIPRec plan-level GRPO
bash scripts/run_experiment.sh --method diprec_plan_grpo --model Qwen/Qwen3-0.6B --dataset "$DATASET" --interest_topk 3 --num_plans 8 --sid_beams 8 --max_history_len 50 --max_seq_len 2048 --seed 42
```

默认参数化方式为 `independent_head`：兴趣 token 使用独立输入 embedding 和输出 head。`--interest_parameterization disjoint_rows` 则让兴趣 token 与 SID token 使用共享矩阵中的不同参数行，可作为原生 vLLM 直接加载模型时的兼容方案。

默认条件模式 `--conditioning interest_bottleneck` 会开启一个新的 SID 解码过程，只向它提供兴趣 plan，不再提供完整历史；`history_visible` 是允许 SID 解码器继续读取历史的消融设置。

为保证数据划分公平，统一实验中的 `sidreasoner` 分支会根据每条历史前缀确定性构造自然语言理由，不混入官方发布的 10-item narrative 语料。它保留 SIDReasoner 的 reasoning→SID SFT/VeRL-GRPO 训练形式，但并非论文 checkpoint 的逐位复现；若要复现原论文设置，可继续使用未改动的原始脚本。

## 批量对比与结果汇总

数据集列表文件只能包含选出的一个或两个数据集。

```bash
# 先运行 seed 42
bash scripts/run_all_comparisons.sh --model Qwen/Qwen3-0.6B --dataset_file configs/selected_long_history_datasets.txt --seeds 42 --max_history_len 50 --max_seq_len 2048 --eval_beams 10 --eval_candidate_budget 80

python scripts/summarize_results.py \
  --input outputs/ \
  --output outputs/comparison.csv

# 仅在 seed 42 上 DIPRec 优于基线后，再补跑三个随机种子
bash scripts/run_all_comparisons.sh --model Qwen/Qwen3-0.6B --dataset_file configs/selected_long_history_datasets.txt --seeds 42,43,44 --max_history_len 50 --max_seq_len 2048 --eval_beams 10 --eval_candidate_budget 80
```

`metrics.json` 和 `comparison.csv` 会记录 Recall@5/10、NDCG@5/10、SID 合法率、兴趣多样性、分层 SID 命中率、评测预算、配置和数据哈希。建议先验证 Qwen3-0.6B；只有小模型结果为正时，再切换到 `Qwen/Qwen3-1.7B`。

## 单独评测 checkpoint

验证集和测试集使用同一个评测器。请根据实际方法和 checkpoint 修改下列路径：

```bash
# 验证集
bash scripts/eval_diprec.sh --method diprec_plan_grpo --model output_dir/$DATASET/history_50/Qwen_Qwen3-0.6B/diprec_plan_grpo/seed_42/final_checkpoint --test_file data/processed/$DATASET/history_50/valid.jsonl --sid_index data/Amazon/index/$DATASET.index.json --output outputs/$DATASET/history_50/Qwen_Qwen3-0.6B/diprec_plan_grpo/seed_42/valid_metrics.json --split valid --max_history_len 50 --max_seq_len 2048 --interest_topk 3 --num_plans 8 --sid_beams 8 --eval_beams 10 --eval_candidate_budget 80 --seed 42

# 测试集
bash scripts/eval_diprec.sh --method diprec_plan_grpo --model output_dir/$DATASET/history_50/Qwen_Qwen3-0.6B/diprec_plan_grpo/seed_42/final_checkpoint --test_file data/processed/$DATASET/history_50/test.jsonl --sid_index data/Amazon/index/$DATASET.index.json --output outputs/$DATASET/history_50/Qwen_Qwen3-0.6B/diprec_plan_grpo/seed_42/metrics.json --split test --max_history_len 50 --max_seq_len 2048 --interest_topk 3 --num_plans 8 --sid_beams 8 --eval_beams 10 --eval_candidate_budget 80 --seed 42
```

完整评测参数可通过以下命令查看：

```bash
bash scripts/eval_diprec.sh --help
```

## 本地检查与尚未验证项

```bash
python -m unittest discover -s tests -v
python -m compileall -q diprec scripts
```

当前已在本地验证：纯 Python 数据与算法契约、原始数据集选择、长历史划分与构建、所有训练/评测入口的 dry-run、Python 语法和 Shell 语法。

当前尚未在本机验证：

- 模型和 tokenizer 的实际下载
- CUDA kernel
- vLLM/VeRL 分布式执行
- GPU 显存与 batch size 调优
- VeRL checkpoint 合并
- 真实数据上的最终指标

`independent_head` 使用仓库提供的 Hugging Face/Accelerate rollout；如果必须让标准 vLLM 引擎直接加载模型，请使用 `disjoint_rows`。
