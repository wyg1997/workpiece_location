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

参数文件的涵义见[configs/config_example.yml](configs/config_example.yml)。

#### 标注工具

```sh
python tools/kps_marking_tool.py \
    --classes [cls1] [cls2] [cls3] ... \
    --path [data_root_path] \
    --img_type ['png', 'jpg', 'bmp'] \
    --idx 1
```

`data_root_path`表示数据的根目录，下面有`source`和`label`两个目录。

#### 训练并测试

```sh
python -m visdom.server -port 8887
CUDA_VISIBLE_DEVICE=1 python tools/train.py \
        --config_file example/baseline/v0/mobilenet_v1.yml \
        VISDOM.PORT 8887
```

#### 单独测试
```sh
python -m visdom.server -port 8887
CUDA_VISIBLE_DEVICE=1 python tools/test.py \
        --config_file example/baseline/v0/mobilenet_v1.yml \
        --checkpoint example/baseline/v0/checkpoint/200.pth \
        VISDOM.PORT 8887
```

### 说明

#### mobilenet_v1

网络结构参考了[Daniil-Osokin/lightweight-human-pose-estimation.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)。

预训练权重来自[sgd_68.848](https://github.com/marvis/pytorch-mobilenet)，使用前需要把参数名字前的`module.`删除掉。删掉后的权重也可以在[我的google网盘下载](https://drive.google.com/file/d/1EYHq40eTpk5FeWxaMrFS4BHxndoZAOKj/view?usp=sharing)。

下载后把预训练模型重命名为`mobilenet_v1.pth`，放在工程根目录的`pretrain/`下。

### 附录

#### commit emoji释义

|    emoji    |     代码      | 释义                                 |
| :---------: | :-----------: | :----------------------------------- |
|   :tada:    |   `:tada:`    | 添加新特性、功能等                   |
|   :fire:    |   `:fire:`    | 删除代码或文件                       |
|   :memo:    |   `:memo:`    | 编写、修改文档                       |
|    :art:    |    `:art:`    | 更改代码格式、结构等不影响功能的问题 |
|    :bug:    |    `:bug:`    | 修复bug                              |
|    :zap:    |    `:zap:`    | 提升速度                             |
| :ambulance: | `:ambulance:` | 修复重要bug                          |
|  :pencil2:  |  `:pencil2:`  | 修改错字                             |
|   :poop:    |   `:poop:`    | 提交低质量代码，需要改进             |
|  :wrench:   |  `:wrench:`   | 修改代码依赖                         |
| :bookmark:  | `:bookmark:`  | 添加tag或release                     |

