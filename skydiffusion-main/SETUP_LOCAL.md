# 在新机器上准备与训练

所有命令均在**包含 `skydiffusion_train.py` 的内层 `skydiffusion-main` 目录**执行。相对路径以当前工作目录为基准，`model_config` 除外（以脚本目录为基准）。不要从外层仓库目录直接运行训练脚本。

## 手动下载的文件放在哪里

```text
skydiffusion-main/                 # 外层 Git 仓库
└── skydiffusion-main/             # 命令运行目录
    ├── skydiffusion_train.py
    ├── clip-vit-large-patch14/
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   ├── vocab.json
    │   ├── merges.txt
    │   ├── tokenizer_config.json
    │   └── special_tokens_map.json
    ├── models/
    │   ├── lacldm_v15.yaml        # Git 已包含
    │   └── v1-5-pruned.ckpt
    ├── configs/g2a3_history.yaml # Git 已包含
    └── datasets/G2A-3/
        ├── History/History/
        │   ├── SVI_before2014_level20_north1/
        │   └── Aerial_before2014_level20/
        └── history_train.csv     # 运行准备脚本生成
```

1. **CLIP：** 从 [openai/clip-vit-large-patch14 文件列表](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) 下载上面列出的六个文件，直接放入 `clip-vit-large-patch14/`，不要再嵌套同名目录。当前环境使用 `pytorch_model.bin`；无需另外下载 `tf_model.h5`、`flax_model.msgpack` 或 `model.safetensors`。已有本地模型时直接复制这六个文件即可。
2. **SD 1.5 初始化权重：** 从 [v1-5-pruned.ckpt 下载页面](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt) 下载文件，放入 `models/`。必须是包含 UNet、VAE 和文本编码器的原始 LDM checkpoint，不能拿单独的 diffusers UNet 文件代替。
3. **G2A-3 History：** 从 [G2A-3 文件列表](https://huggingface.co/datasets/Yejy53/G2A-3/tree/main) 下载 `History.zip` 并解压到 `datasets/G2A-3/` 下。准备脚本会递归查找 `SVI_before2014_level20_north1`，允许不同解压嵌套层级；其同级必须有 `Aerial_before2014_level20`。当前配置只训练 History，暂时不需要 UAV 和 disaster。

下载 SD 权重也可执行（PowerShell，支持断点续传）：

```powershell
curl.exe --location --fail --retry 5 --continue-at - --output ./models/v1-5-pruned.ckpt "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt?download=true"
```

## 准备和运行

从外层仓库进入内层目录，首次创建环境：

```powershell
cd skydiffusion-main
conda env create -f environment.yaml
conda activate skydiffusion
```

已有环境时只需激活。模型与数据放好后执行：

```powershell
python prepare_g2a3.py
python skydiffusion_train.py --config configs/g2a3_history.yaml --check-data
python skydiffusion_train.py --config configs/g2a3_history.yaml --sd-checkpoint models/v1-5-pruned.ckpt --smoke-test --output-dir training_runs/g2a3_smoke
```

两步测试成功后，正式训练：

```powershell
python skydiffusion_train.py --config configs/g2a3_history.yaml --sd-checkpoint models/v1-5-pruned.ckpt
```

训练清单 `datasets/G2A-3/history_train.csv` 中的图片路径相对于配置中的 `data_root`。每台机器解压后重新运行准备脚本，不需要手工编辑 CSV。CLIP 使用本地目录加载，无需通过代理下载。训练输出在 `training_runs/g2a3_history/`。

## 如果文件必须放在其他磁盘

无需修改代码，覆盖命令行参数即可。下面的路径仅为示例，请替换为实际位置；Linux 使用对应的 `/mnt/...` 路径：

```powershell
python prepare_g2a3.py --data-root "D:/datasets/G2A-3"
python skydiffusion_train.py --config configs/g2a3_history.yaml --data-root "D:/datasets/G2A-3" --clip-model "D:/weights/clip-vit-large-patch14" --sd-checkpoint "D:/weights/v1-5-pruned.ckpt"
```

准备脚本与训练必须使用相同的数据根目录。也可修改 YAML 的 `data_root` 和 `clip_model`；SD 权重通过 `--sd-checkpoint` 指定。遵循上面的默认目录布局则不用改任何路径。

## 验证范围与 Git

当前正式参数为 512×512、FP32、batch size 1、梯度累积 4、20000 步。4GB 显存、16GB 内存的本机未完成完整模型训练，缩小输入尺寸也不能消除权重占用。已通过 22 项测试，以及真实 History 图片的小模型 GPU 两步训练、梯度更新和检查点保存；这不等同于完整 SD 1.5 模型训练验证。新机器请先运行上述 smoke test。

History 暂用现有 CVACT 投影参数作近似，尚未针对 G2A-3 标定，也没有独立验证集。

`.gitignore` 已忽略 CLIP 目录、模型权重、数据集、生成清单、缓存和训练输出；保留模型 YAML、代码及文档。小模型测试产生的 checkpoint 不能作为正式 SkyDiffusion 初始化权重。
