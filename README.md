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

#### 设置参数

参数文件的涵义见`configs/example_config.yml`。

#### 标注工具

```sh
python tools/kps_marking_tool.py \
    --classes [cls1] [cls2] [cls3] ... \
    --path [data_root_path] \
    --img_type ['png', 'jpg', 'bmp'] \
    --idx 1
```

#### 训练并测试

```sh
python -m visdom.server -port 8887
CUDA_VISIBLE_DEVICE=1 python tools/train.py --config_file example/baseline/v0/resnet34.yml VISDOM.PORT 8887
```

