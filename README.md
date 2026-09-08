# SIDReasoner

> DIPRec 七组对比实验的精简步骤与命令见 [MY_README_ZH.md](MY_README_ZH.md)；English guide: [MY_README.md](MY_README.md). 七模型使用独立 Python 3.11 + TRL 0.24 环境；下方原 SIDReasoner/VeRL 文档及其环境要求保持不变，并与七模型环境分开。

> **DIPRec RL 修正说明：** `diprec_traj_rl` / `diprec_plan_rl` 已迁移至 TRL 0.24：使用固定 DIPRec-SFT reference KL、复用 rollout 的多轮更新和有效 PPO clipping；`diprec_plan_rl` 仍保留 plan/SID 两级 advantage。

> **MiniOneRec RL 奖励实验：** `main_miss_prefix` 的 1,000 步 A/B 已完成，λ=0.1 的辅助信号已启用，但尚未建立推荐收益；默认仍使用原奖励。复现命令见 [中文指南](MY_README_ZH.md#3-rl)，结果与下一步建议见 [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)。

> **串行验证已完成：** 原奖励和 λ=0.3/0.5/1.0 四组均完成 1,000 步，增强前缀信号仍未建立可信推荐收益，当前停止单纯扫描系数。[run_prefix_sweep.py](scripts/run_prefix_sweep.py) 保留串行运行与汇总功能；结果和后续方向见 [实验历史](EXPERIMENT_HISTORY.md#13-已完成串行前缀强度验证)。

> **五组 RL 优化已有部分结果（2026-09-09）：** A fixed、B sync 均完成3,455步，未提高Valid排名；固定train/valid探针的目标相对错误候选margin均下降。C在约第60步因磁盘不足中断，D/E未开始，降低β和提高LR尚无结论。释放训练机器磁盘后用 `--resume` 复用A/B、重跑C并执行D/E。配置与命令见 [中文指南](MY_README_ZH.md#4-五组-rl-优化实验部分完成)，结果见 [实验历史](EXPERIMENT_HISTORY.md#141-2026-09-09-上传结果ab完成c磁盘不足中断)。

> **额外 F 组（已实现，待运行）：** [run_rl_extra_experiment.py](scripts/run_rl_extra_experiment.py) 使用4–7卡与独立端口，关闭KL惩罚（β=0），其余沿用A；复用同一SFT与冻结探针，独立保存状态和三个阶段结果，可与0–3卡的C/D/E并行。启动前先解决已有磁盘不足问题，命令见[中文指南](MY_README_ZH.md#额外f组使用4567卡)。

This is the code implementation for **"SIDReasoner - Reasoning over Semantic IDs Enhances Generative Recommendation"**.

SIDReasoner is a generative recommendation framework that strengthens generative recommenders with reasoning ability over semantic IDs. This repository provides:

- A complete training pipeline, with each training stage integrated into an easy-to-run script.
- Full training data, including our synthesized enriched alignment corpus.
- Pretrained model checkpoints.

Our method demonstrates that, with improved SID–language alignment, effective recommendation reasoning can be achieved even under academic-scale training. SIDReasoner is able to associate SIDs with their underlying item semantics, produce coherent natural-language reasoning over interaction histories, and generate recommendations according to the reasoning process. By open-sourcing the pipeline, data, and checkpoints, we aim to facilitate further research on reasoning in generative recommendation.

<p align="center">
  <img src="assets/SIDReasoner-CaseStudy.png" width="100%" alt="Training">
</p>
<p align="center">
  <em>A case study of how SIDReasoner generates interpretable reasoning over SIDs.</em>
</p>

## Environments

The reinforcement learning stage (Stage 3) in this project is built on top of VERL. We recommend follow the [official installation guide](https://verl.readthedocs.io/en/latest/start/install.html#requirements) to set up the environment. To execute the codes correctly, the following additional packages are required:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `pandas`
- `numpy`
- `fire`
- `wandb`
- `tqdm`
- `accelerate`
- `bitsandbytes`


## Dataset

The datasets can be accessed via this [link](https://drive.google.com/file/d/1etg1e8oStGOjsg1Vr15vFnjlTMUx4Htz/view?usp=sharing). Please download the dataset and ensure the dataset folder is placed under directory ./data/Amazon .

## Training

SIDReasoner follows a three-stage training pipeline.

| Stage | Script | 
| --- | --- | 
| Stage 1: Supervised Fine-Tuning | `bash sft_Qwen3_enrich.sh` | 
| Stage 2: Reasoning Activation | `bash sft_reasoning_activation.sh` |
| Stage 3: RL Training | `bash RL_training_script.sh` |

### Run training

```bash
# Stage 1
bash sft_Qwen3_enrich.sh

# Stage 2
bash sft_reasoning_activation.sh

# Stage 3
bash RL_training_script.sh
```

The training logs are written to `./logs`.


### Checkpoints

To facilitate further research, we release our pretrained model checkpoints, which can be downloaded via this [link](https://huggingface.co/Sober-Clever/SIDReasoner-Models/tree/main).

## Evaluation

We provide the scripts to test the model performance under thinking and non-thinking mode:

```bash
# Non-thinking mode.
bash evaluate_Qwen3.sh

# Thinking mode.
bash evaluate_Qwen3_think.sh
```

### Stage 3 checkpoint merge

The reasoning evaluation script expects a merged Hugging Face checkpoint named `actor_merged`. If RL training has only produced raw `actor` folders, merge them first:

```bash
python3 ./scripts/merge_fsdp_checkpoint.py \
  --checkpoint ./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor \
  --output-dir ./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged
```


## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{SIDReasoner,
  title={Reasoning over Semantic IDs Enhances Generative Recommendation},
  author={Yingzhi He and Yan Sun and Junfei Tan and Yuxin Chen and Xiaoyu Kong and Chunxu Shen and Xiang Wang and An Zhang and Tat-Seng Chua},
  journal={arXiv preprint arXiv:2603.23183},
  year={2026}
}
```

## Acknowledgement

This repo is built upon [MiniOneRec](https://github.com/AkaliKong/MiniOneRec). 
