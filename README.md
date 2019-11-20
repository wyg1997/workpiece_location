# 关键点检测

### 代码目录结构

```text
--| config/     -> 默认参数
--| configs/    -> 可用的参数文件
--| datasets/   -> 数据处理
--| engine/     -> 训练、推理引擎
--| models/     -> 内置网络结构
--| solver/     -> 优化器、学习率调整器
--| tools/      -> 训练、推理脚本
--| utils/      -> 工具包
```

### Usage

```sh
CUDA_VISIBLE_DEVICE=1 python tools/train.py --config_file configs/resnet34.yml VISDOM.PORT 8887
```
