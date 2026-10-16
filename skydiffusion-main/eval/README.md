# 历史评估脚本

本目录保留上游的研究与数据处理脚本，不是已验证的统一评估入口，也不是训练或示例推理的必需依赖。

`eval.py` 接收预测目录与真值目录两个位置参数，要求同名配对图片。历史脚本的本机绝对路径已改为相对示例路径，统一从内层项目目录运行（例如 `python eval/eval.py ./results/fake ./results/real`）。`datasets/` 尚无数据，其他脚本的目录和文件名仍需按实际数据调整；相对路径占位不代表数据已经准备好。不要直接批量运行本目录脚本。

除了主环境，部分脚本还导入 `lpips`、`pytorch-fid`、`pytorch-ssim`、`scikit-image`。这些可选依赖尚未统一锁定和验证。`eval.py` 使用旧版 `skimage.measure.compare_ssim/compare_psnr`，不能仅安装最新版 scikit-image 就认为兼容；迁移时需要核对数据范围、通道轴和指标定义。

`eval.py` 中 LPIPS 实际计算被注释，导入 FID 库也不代表计算了 FID；当前输出包括 PSNR、SSIM 和名为 L1 的梯度差指标。请勿把这些脚本的输出直接视为论文的完整复现结果。正式评估前需要单独验证指标实现、依赖和配对规则。
