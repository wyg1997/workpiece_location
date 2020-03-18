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
output_template=>operation: 返回匹配好的模板
output_keypoints=>operation: 每个关键点作为独立的模板返回
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

### 模板匹配

#### 输入

- 网络计算出的关键点，有坐标(角度、半径)、类别信息。
- 模板的实例化对象，有各关键点的坐标(角度、半径)、类别以及匹配得分阈值、旋转角度范围、缩放范围等信息。

#### 相关公式

1. 仿射变换：

   $$\begin{pmatrix}1 & 0 & tx \\ 0 & 1 & ty \\ 0 & 0 & 1\end{pmatrix}*\begin{pmatrix} cos\theta & -sin\theta & 0 \\ sin\theta & cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix} * \begin{pmatrix} sx & 0 & 0 \\ 0 & sy & 0 \\ 0 & 0 & 1 \end{pmatrix} * \begin{pmatrix} 1 & shearX & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} * \begin{pmatrix} X(x_0, x_1 ...) \\ Y(y_0, y_1 ...) \\ I(1, 1 ...) \end{pmatrix} = \begin{pmatrix} X' \\ Y' \\ I \end{pmatrix}$$

   从左到右一共6个矩阵，分别表示：

   1. 平移变换。
   2. 旋转变换。
   3. 缩放(`sx`表示x轴缩放倍数，`sy`表示y轴缩放倍数)。
   4. x错切变换。
   5. 变换前的坐标。
   6. 变换后的坐标。

   所以图像变换的顺序应该为`x错切 -> 缩放 -> 旋转 -> 平移`。

   > 注意：错切变切只能出现一个，所以这里只用x轴错切。

2. 化简：

   $$\begin{pmatrix} sx*cos\theta & shearX*sx*cos\theta-sy*sin\theta & tx \\ sx*sin\theta & shearX*sx*sin\theta+sy*cos\theta & ty \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} a & c & e \\ b & d & f \\ 0 & 0 & 1 \end{pmatrix}$$

3. 正反推：

   $$\left \{ \begin{array}{ll} a=sx*cos\theta \\ b=sx*sin\theta \\ c=shearX*sx*cos\theta-sy*sin\theta \\ d=-sy*cos\theta \\ e=tx \\ f=ty \end{array} \right.$$

   $$\left\{ \begin{array}{ll} angle=atan2(b, a) \\ denom(assist)=a^2+b^2 \\ sx=sqrt(denom) \\ sy=(a*d-c*b)/(sx+eps) \\ shearX=atan2(a*c+b*d, denom) \\ tx=e \\ ty=f \end{array}\right.$$

> 有了上述公式，我们可以通过变换前后的坐标得到仿射变换矩阵，也可以通过仿射变换矩阵得到变换的参数(旋转角、缩放倍数、平移量、错切量)。

#### 匹配流程

1. dfs按每个点的类别得到所有点的组合，组成候选列表（包括部分点不存在的情况）。
2. 求出模板相对于每组候选点的仿射变换参数，排除过于夸张的变换（错切量过大、x和y的缩放比过于夸张）以及单点的属性无法对应的组合。并对组合进行扣分处理，总分为1.0，缺一个点扣`1/n_node`分，位置、角度、半径的误差也进行扣分。取出得分在阈值以上的组合，保存`组合列表` `仿射变换矩阵` `仿射变换参数` `得分`。
3. 以得分为关键字进行排序，由高到低组合成模板，每个点只能匹配一次。

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