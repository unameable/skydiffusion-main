# 待下载的数据集

此目录用于存放之后下载的数据，目前只有目录占位文件，没有数据集或训练清单。
下载入口见 [README 的数据准备章节](../README.md#data-preparation)。

```text
datasets/
├── G2A-3/
├── CVUSA/
├── CVACT/
└── VIGOR/
```

请把压缩包与解压内容放进对应子目录。解压后根据真实配对信息创建 `train.csv`、`val.csv`；例如 CVACT 的清单放在 `datasets/CVACT/`，清单里的图片路径相对于该清单所在目录。目录内的名称不代表原始数据包结构。

所有命令从内层项目目录（包含 `skydiffusion_train.py` 的目录）运行。数据准备完成后检查：

```powershell
python skydiffusion_train.py --train-data ./datasets/CVACT/train.csv --val-data ./datasets/CVACT/val.csv --dataset-name CVACT --check-data
```

当前尚未下载数据，因此上述命令暂时不能执行。模型权重按 [models/README.md](../models/README.md) 单独准备。

Git 只保留本说明与 `.gitkeep` 占位文件，下载的压缩包、图片和清单均被忽略。
