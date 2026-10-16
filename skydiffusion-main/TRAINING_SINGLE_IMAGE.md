# 单幅街景 → 航拍图训练

本训练入口使用现有 `LAControlLDM + LAControlNet + ControlledUnetModel`，每个样本只有一张街景和一张对应航拍图。训练入口、权重映射和参数设置位于 `skydiffusion_train.py`、`skydiffusion_training.py`、`configs/single_image.yaml`。不包含 Multi-to-One。

## 环境与数据

在项目内层目录（包含 `skydiffusion_train.py` 的目录）运行。沿用仓库的 Python 3.8 / PyTorch 1.12.1 / pytorch-lightning 1.5.0 环境。例如 PowerShell：

```powershell
conda activate skydiffusion
python skydiffusion_train.py --help
```

一行表示一个配对样本。推荐使用有表头的 CSV 或 TSV，列名固定为 `name,source,target,prompt`，其中 `prompt` 可省略：

```csv
name,source,target,prompt
sample_001,street/001.jpg,aerial/001.png,
sample_002,street/002.jpg,aerial/002.png,
```

`source` 是 RGB 全景街景，`target` 是同一位置且朝向对齐的 RGB 航拍图。路径可以是绝对路径；相对路径优先相对于清单所在目录，或使用 `--data-root` 指定统一根目录。无表头旧格式也保留：CVACT 为 `name<TAB>source<TAB>target[<TAB>prompt]`；旧 CVUSA 三列为 `name,target,source`，需传 `--is-cvusa`。新数据推荐有表头格式，避免列顺序歧义。

自行准备互不重叠的 `train.csv`、`val.csv`；同一地点的相邻街景应放在同一划分，避免位置泄漏。程序不会随机拆分地理位置，也不会估计缺失的朝向信息。

`--source-mode panorama`（默认）会执行 Curved-BEV，几何参数目前针对 CVACT/CVUSA。`--source-mode bev` 可直接读取已经对齐的 RGB BEV 条件图（含自定义数据），跳过几何变换。`.pkl` 作为已经转换的 BEV，必须保存 HWC RGB 的浮点 `[0,1]` 或 uint8 `[0,255]` 数组。模型最终接收 `hint=[0,1]`、`jpg=[-1,1]`。

```powershell
python skydiffusion_train.py --train-data ./datasets/CVACT/train.csv --val-data ./datasets/CVACT/val.csv --check-data
```

检查 `training_runs/single_image/data_preview`：每张预览左侧为 BEV，右侧为 GT；`ranges.json` 记录前四个样本的形状和范围。这是小样本检查，不代替全量数据清洗。

**与原仓库预处理的变化：** 修复 BEV 已在 `[0,1]` 却再次除以 255；旋转参数明确为角度，默认 `--rotation-degrees 90`，按 `π/180` 换算。旧代码的 `90π` 模 `2π` 约为零；若要检查旧权重的历史方向，可显式使用 `--rotation-degrees 0`。请根据预览与数据集朝向选择，并在训练和推理中保持一致。普通透视照片不能直接当作全景图输入。

## 权重初始化与训练

三种权重参数互斥，正式训练必须指定一种：

1. `--sd-checkpoint`：原始 LDM 格式的 Stable Diffusion 1.5 checkpoint，包含 `model.diffusion_model.*`、`first_stage_model.*`、`cond_stage_model.*`。不接受单独的 diffusers UNet、SD2/SDXL 或残缺权重。程序复制 SD 编码器/中间层到控制分支，将控制首层从 4 通道扩为 8 通道，前四个噪声通道复制 SD 权重，后四个条件通道填零，保留零卷积。条件编码器从已加载 VAE 权重同步。映射方式参考 [ControlNet 官方初始化脚本](https://github.com/lllyasviel/ControlNet/blob/main/tool_add_control.py)，并增加本项目所需的输入扩展和条件编码器映射。
2. `--init-checkpoint`：完整的 SkyDiffusion 权重，开始新的优化器和步数；所有模型键都必须存在并且形状兼容。
3. `--resume`：此训练入口保存的 Lightning checkpoint，恢复模型、优化器及全局步数。恢复时使用同一训练配置和相同参数冻结设置。

CLIP tokenizer 和文本模型默认使用 `openai/clip-vit-large-patch14`，首次会由 transformers 下载。离线时用 `--clip-model ./clip-vit-large-patch14`，该目录必须包含 tokenizer、配置和模型权重。

先用真实数据/权重跑两步验证：

`--smoke-test` 用于开始新的两步运行，与 `--resume` 互斥；检查恢复训练时，使用高于已保存步数的 `--max-steps`。

```powershell
python skydiffusion_train.py --train-data ./datasets/CVACT/train.csv --val-data ./datasets/CVACT/val.csv --sd-checkpoint ./models/v1-5-pruned.ckpt --clip-model ./clip-vit-large-patch14 --smoke-test --output-dir training_runs/smoke
```

正式训练示例：

```powershell
python skydiffusion_train.py --train-data ./datasets/CVACT/train.csv --val-data ./datasets/CVACT/val.csv --sd-checkpoint ./models/v1-5-pruned.ckpt --clip-model ./clip-vit-large-patch14 --precision 16 --batch-size 1 --accumulate-grad-batches 4 --max-steps 20000 --output-dir training_runs/cvact_single
```

`--dataset-name CVUSA` 选择 CVUSA 几何参数。对已经转换的 BEV，加入 `--source-mode bev`。已有 SkyDiffusion 权重时，将 `--sd-checkpoint` 换成 `--init-checkpoint ./models/CVACT_Skydiffusion.ckpt`。

默认只训练控制网络，VAE、条件编码器、文本编码器和 SD 主干冻结；冻结的 SD 解码器仍保留对控制特征的梯度传播。增加 `--train-unet-decoder` 可额外训练 U-Net 解码器和输出层，编码器/中间层仍冻结。学习率默认 `1e-5`，使用原有噪声预测 L2 损失和 AdamW，默认不使用 EMA。

完整模型包含 SD 主干、控制分支、VAE、CLIP 和优化器状态，4 GB 显存通常无法完成 512×512 训练。梯度累积增加有效批大小，但无法消除单个样本的显存需求。可用 `--image-size 256 --precision 16` 探索较低分辨率，正式训练应准备显存更充足的设备；无法保证这个调整能在 4 GB 上运行。CPU 模式使用 `--accelerator cpu --precision 32`，完整模型很慢。本地 Windows 默认单 GPU，Linux 多 GPU 使用 `--devices 2` 等。

## 输出、恢复与推理

- `checkpoints/step-*.ckpt`：默认每 1000 个优化器步保存一次；`last.ckpt` 用于恢复。完整检查点包含优化器状态，体积较大，请按磁盘容量设置 `--checkpoint-every`。
- `training_config.yaml`、`model_config.yaml`：解析后的训练与模型配置。
- `initialization.json`：权重加载和训练参数数量报告。关键参数缺失或形状不匹配会直接报错。
- `metrics/version_*/metrics.csv`：训练/验证损失。
- `image_log/`：默认每 500 步生成一个样本及 BEV、重建、文本预览；`--image-log-every 0` 可关闭。训练预览使用引导系数 1，主要用于检查是否学习了条件对应关系。

恢复（`max_steps` 是总目标步数，不是追加步数）：

Lightning 1.5 会恢复模型、优化器和全局步数，但默认不精确恢复 epoch 内的数据读取位置，因此不保证逐位复现中断前的采样顺序。

```powershell
python skydiffusion_train.py --config training_runs/cvact_single/training_config.yaml --resume training_runs/cvact_single/checkpoints/last.ckpt --max-steps 40000
```

使用保存的模型配置推理，并保持相同 BEV 设置：

```powershell
python skydiffusion_test.py --num_gpus=1 --batch_size=1 --config_path=training_runs/cvact_single/model_config.yaml --model_path=training_runs/cvact_single/checkpoints/last.ckpt --data_file_path=./datasets/CVACT/val.csv --result_dir=results/cvact_single --dataset_name=CVACT --source_mode=panorama --rotation_degrees=90
```

若训练时用的是 `source_mode=bev` 或其他旋转角度，推理也相应设置。当前测试入口仍需要配对清单中的 GT。

## 开发验证

```powershell
python -m unittest discover -s tests -p "test_single_image*.py" -v
```

小模型测试验证数据范围、初始化映射和实际梯度/保存恢复，不需要下载完整 SD/CLIP。通过小模型测试不代表完整模型已收敛或达到了论文指标；正式训练先检查实际数据预览与两步运行，再用独立验证集评估。
