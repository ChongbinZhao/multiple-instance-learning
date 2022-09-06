### 提示：该README.md所用到的图片都保存在pictures文件夹里，图片没法加载的话可能需要你先挂一下vpn！
<br></br>


#### Attention-based Deep Multiple Instance Learning 论文复现

<br>

**论文链接：** https://arxiv.org/abs/1802.04712

<br>

**数据集说明：**

- 论文作者用来训练`Attention MIL`模型用到了两个数据集，分别是`MNIST`手写数字数据集`COLON CANCER`数据集，论文作者在他的`Github`项目没有提供，大家可以从我的网盘[这里](https://pan.baidu.com/s/1pX2ddManpNUZjwkW4IEWhQ?pwd=6666  )获取，提取码是6666。
- 其中`COLON CANCER`数据集包含处理过和未处理过的样本：
  - 原`COLON CANCER`数据集包含100张500×500像素的slide。
  - 按论文的要求处理后，每张500×500像素的slide被切割成若干张27×27的patches（论文要求白像素点超过75%的patch要丢弃），训练模型的时候直接用处理过的样本进行训练即可。
- 数据集下载后，将两个数据集分别放到对应的文件目录下，在相应的main.py文件里修改一下路径，然后运行即可。

<br>

**代码说明:**

- 论文作者在Github上只提供了训练`MNIST`数据集的代码，而没有提供训练`COLON CANCER`数据集的代码。
- 论文中对于不同的数据集，网络结构不同、超参数不同以及训练方式不同，所以不能将`MNIST`数据集的代码直接套用在`COLON CANCER`数据集上。
- 于是我根据论文提供的思路以及神经网络的参数，自己重头写了一遍训练`COLON CANCER`数据集的代码，最后训练出来的指标与论文结果保持一致，所以我的代码应该是没有错的。

<br>

**原理讲解：**

我将自己对论文的理解以及实现过程通过word笔记的方式记录下来了（我都贴在下面了），我的笔记可能讲得不是很清楚 ，强烈建议先去看几遍原论文。

<br>

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/1.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/2.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/3.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/4.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/5.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/6.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/7.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/8.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/9.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/10.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/11.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/12.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/13.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/14.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/15.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/16.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/17.jpg)

![img](https://raw.githubusercontent.com/ChongbinZhao/multiple-instance-learning/master/pictures/18.jpg)
