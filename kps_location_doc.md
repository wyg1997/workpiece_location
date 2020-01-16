# 定位模块

## 目录

{:toc}

## 功能

- 位置预测
- 多类别预测
- 角度预测(可选)
- 半径预测(可选)
- 模板匹配(可选)

## 算法流程

### 训练

```flow
st=>start: start
model_loader=>operation: 对应模型加载
data_loader=>operation: 数据载入
data_augmentation=>operation: 数据增强(更改尺寸、翻转、缩放、旋转、光照)
create_label=>operation: 生成标签
train=>operation: 训练
save_model=>operation: 保存模型
e=>end: end

st->model_loader->data_loader
data_loader->data_augmentation->create_label->train
train->save_model->e
```

### 推理

```flow
st=>start: start
model_loader=>operation: 模型加载
template_loader=>operation: 模板加载
data_loader=>inputoutput: 数据加载
single_test=>operation: 推理一次
get_keypoints=>operation: 从网络输出中拿到关键点信息
have_template=>condition: 是否存在模板
template_match=>operation: 模板匹配
output_template=>inputoutput: 返回匹配好的模板
output_keypoints=>inputoutput: 每个关键点作为独立的模板返回
finish_single_test=>operation: 完成一次推理
is_test_all=>condition: 推理完成
e=>end: end

st->model_loader->template_loader->data_loader->single_test
single_test->get_keypoints->have_template
have_template(yes)->template_match->output_template->finish_single_test
have_template(no)->output_keypoints->finish_single_test
finish_single_test->is_test_all
is_test_all(no)->single_test
is_test_all(yes)->e
```

## 界面需求

### 标注界面

- 标注界面应该有`radius`和`angle`两个开关选择是否对半径和角度进行预测。
- 标注时根据上述两个开关对半径和角度进行调整。
- 标注的点可以设置类别。
- 标注点的样式设计。

### 模板定义界面

- 可以定义多个模板
- 有添加、删除按钮来对当前模板的节点进行操作。
- 模板的参数和每个点的参数设置区域。