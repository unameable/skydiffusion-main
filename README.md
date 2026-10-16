# SkyDiffusion — single-image training

本仓库基于 SkyDiffusion，包含单幅街景到航拍图的训练入口、数据检查、权重初始化和断点恢复功能。原论文与作者信息见[项目说明](skydiffusion-main/README.md)。

代码位于内层 `skydiffusion-main/`，请从仓库根目录进入该目录后运行命令：

```powershell
cd skydiffusion-main
conda env create -f environment.yaml
conda activate skydiffusion
python skydiffusion_train.py --help
```

- [安装、示例推理与数据准备](skydiffusion-main/README.md)
- [单幅训练、数据格式和恢复训练](skydiffusion-main/TRAINING_SINGLE_IMAGE.md)
- [模型权重准备](skydiffusion-main/models/README.md)
- [评估脚本的依赖与限制](skydiffusion-main/eval/README.md)
- [Apache 2.0 许可证](skydiffusion-main/LICENSE)

仓库包含代码、配置、测试及少量示例图片；完整数据集、模型权重、日志和运行结果不随 Git 发布。模型和第三方组件保留各自的许可证。

当前验证覆盖小模型训练、梯度、保存恢复和示例数据读取，不代表完成真实权重的完整训练、论文指标复现或全新环境安装验证。

## 数据集下载

G2A-3 支持命令行下载：见[PowerShell 下载、断点续传及解压命令](skydiffusion-main/README.md#命令行下载-g2a-3windows-powershell)。也提供了浏览器下载备用方法。命令将文件保存到项目的 `datasets/G2A-3/` 目录。

| 数据集 | 下载入口 | 说明 |
| --- | --- | --- |
| G2A-3 | [Hugging Face 文件列表](https://huggingface.co/datasets/Yejy53/G2A-3/tree/main) | 包含 `History.zip`、`UAV.zip`、`disaster.zip`，可按需下载 |
| CVUSA | [官方数据集页面](https://mvrl.cse.wustl.edu/datasets/cvusa/) | 按官方说明下载街景与卫星配对数据 |
| CVACT | [作者项目 OriCNN](https://github.com/Liumouliu/OriCNN#act-dataset) | 按作者说明申请 ACT_small / ACT_test 数据 |
| VIGOR | [官方项目](https://github.com/Jeff-Zilence/VIGOR) | 按项目 README 中的数据下载说明获取 |

下载与整理步骤、G2A-3 压缩包直链，以及本训练入口支持的数据格式，见[详细数据准备说明](skydiffusion-main/README.md#data-preparation)。已创建 [datasets/](skydiffusion-main/datasets/README.md) 目录及各数据集子目录，供之后下载使用；目前没有下载数据。所有运行命令中的相对路径均以内层项目目录为基准。
