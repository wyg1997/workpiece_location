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

### 说明

#### mobilenet_v1

网络结构参考了[Daniil-Osokin/lightweight-human-pose-estimation.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)。

预训练权重来自[sgd_68.848](https://github.com/marvis/pytorch-mobilenet)，使用前需要把参数名字前的`module.`删除掉。

删掉后的权重也可以在[我的google网盘下载](https://drive.google.com/file/d/1EYHq40eTpk5FeWxaMrFS4BHxndoZAOKj/view?usp=sharing)。

