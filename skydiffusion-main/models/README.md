# 模型权重准备

以下路径均相对于内层项目目录。权重不纳入 Git；请自行下载或通过训练生成。

## CLIP 文本编码器

训练配置默认使用本地相对路径 `./clip-vit-large-patch14`，请先手动准备下面的文件。完整迁移步骤见 [新机器准备说明](../SETUP_LOCAL.md)。推理入口仍可通过 `--clip_model` 指定本地目录。
文件来源：[OpenAI CLIP 模型仓库](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)。

离线使用时，把以下文件放到 `clip-vit-large-patch14/`：

- `config.json`
- `pytorch_model.bin`（匹配本项目 transformers 4.19.2）
- `vocab.json`、`merges.txt`
- `tokenizer_config.json`、`special_tokens_map.json`

不需要同时下载 TensorFlow、Flax 和 safetensors 三种额外权重。
训练时传 `--clip-model ./clip-vit-large-patch14`；推理时传 `--clip_model ./clip-vit-large-patch14`。

## 从 Stable Diffusion 1.5 初始化训练

可从 [SD 1.5 文件仓库](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) 获取原始 LDM 格式 checkpoint，例如 `v1-5-pruned.ckpt`，放到 `models/v1-5-pruned.ckpt`。
需要完整的 UNet、VAE 和文本编码器权重；不能用 diffusers 的单独 UNet 文件代替。

```powershell
python skydiffusion_train.py --train-data ./datasets/CVACT/train.csv --val-data ./datasets/CVACT/val.csv --sd-checkpoint ./models/v1-5-pruned.ckpt --smoke-test
```

`datasets/CVACT/train.csv` 和 `datasets/CVACT/val.csv` 由使用者准备，格式见 [训练说明](../TRAINING_SINGLE_IMAGE.md)。SD 初始化权重不是训练完成的 SkyDiffusion 推理权重。

