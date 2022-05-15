---
title: Self-Attention机制
tags: [Self-Attention, NLP, Transformer]
---

<meta name="referrer" content="no-referrer" />

NLP领域Transformer模型中self-attention机制

<!--more-->

# 前言

注意力模型最近几年在深度学习各个领域被广泛使用，无论是图像处理、语音识别还是自然语言处理的各种不同类型的任务中，都很容易遇到注意力模型的身影。

从注意力模型的命名方式看，很明显其借鉴了人类的注意力机制:视觉注意力机制是人类视觉所特有的大脑信号处理机制。人类视觉通过快速扫描全局图像，获得需要重点关注的目标区域，也就是一般所说的注意力焦点，而后对这一区域投入更多注意力资源，以获取更多所需要关注目标的细节信息，而抑制其他无用信息。这是人类利用有限的注意力资源从大量信息中快速筛选出高价值信息的手段，是人类在长期进化中形成的一种生存机制，人类视觉注意力机制极大地提高了视觉信息处理的效率与准确性。

深度学习中的注意力机制从本质上讲和人类的选择性视觉注意力机制类似，核心目标也是从众多信息中选择出对当前任务目标更关键的信息。

# Encoder-Decoder框架

Encoder-Decoder框架可以看作是一种深度学习领域的研究模式，应用场景异常广泛。目前大多数注意力模型附着在Encoder-Decoder框架下，当然，其实注意力模型可以看作一种通用的思想，本身并不依赖于特定框架.

文本处理领域的Encoder-Decoder框架可以这么直观地去理解：可以把它看作适合处理由一个句子（或篇章）生成另外一个句子（或篇章）的通用处理模型。对于句子对<Source,Target>，我们的目标是给定输入句子Source，期待通过Encoder-Decoder框架来生成目标句子Target。Source和Target可以是同一种语言，也可以是两种不同的语言。

Source和Target分别由各自的单词序列构成：


$$
Source = <x_1,x_2,...,x_m> \\
Target=<y_1,y_2,...,y_n>
$$


Encoder顾名思义就是对输入句子Source进行编码，将输入句子通过非线性变换转化为中间语义表示C：
$$
C=\mathcal{F}(x_1,x_2,...,x_m)
$$
对于解码器Decoder来说，其任务是根据句子Source的中间语义表示C和之前已经生成的历史信息$y_1,y_2,...,y_{i-1}$来生成i时刻要生成的单词$y_i$:


$$
y_i=\mathcal{G}(C,y_1,y_2,...,y_{i-1})
$$


每个$y_i$都依次这么产生，那么看起来就是整个系统根据输入句子Source生成了目标句子Target。如果Source是中文句子，Target是英文句子，那么这就是解决机器翻译问题的Encoder-Decoder框架；如果Source是一篇文章，Target是概括性的几句描述语句，那么这是文本摘要的Encoder-Decoder框架；如果Source是一句问句，Target是一句回答，那么这是问答系统或者对话机器人的Encoder-Decoder框架。由此可见，在文本处理领域，Encoder-Decoder的应用领域相当广泛。

![image-20220515135444239](https://gitee.com/Rien_190/img-url/raw/master/img/202205151717314.png "图1 抽象的文本处理领域的Encoder-Decoder框架")

<center><p>图1 抽象的文本处理领域的Encoder-Decoder框架</p></center>

Encoder-Decoder框架不仅仅在文本领域广泛使用，在语音识别、图像处理等领域也经常使用。比如对于语音识别来说，图1所示的框架完全适用，区别无非是Encoder部分的输入是语音流，输出是对应的文本信息；而对于“图像描述”任务来说，Encoder部分的输入是一副图片，Decoder的输出是能够描述图片语义内容的一句描述语。一般而言，文本处理和语音识别的Encoder部分通常采用RNN模型，图像处理的Encoder一般采用CNN模型。

# Attention模型

本节先以机器翻译作为例子讲解最常见的Soft Attention模型的基本原理，之后抛离Encoder-Decoder框架抽象出了注意力机制的本质思想，然后简单介绍最近广为使用的Self Attention的基本思路。

## Soft Attention模型

图1中展示的Encoder-Decoder框架是没有体现出“注意力模型”的，所以可以把它看作是注意力不集中的分心模型。为什么说它注意力不集中呢？请观察下目标句子Target中每个单词的生成过程如下：


$$
\begin{align}
y_1=&f(C)\\
y_2=&f(c,y_1)\\
y_3=&f(c,y_1,y_2)
\end{align}
$$


其中f是Decoder的非线性变换函数。从这里可以看出，在生成目标句子的单词时，不论生成哪个单词，它们使用的输入句子Source的语义编码C都是一样的，没有任何区别。

而语义编码C是由句子Source的每个单词经过Encoder 编码产生的，这意味着不论是生成$y_1,y_2,y_3$哪个单词,其实句子Source中任意单词对生成某个目标单词$y_i$来说影响力都是相同的，这是为何说这个模型没有体现出注意力的缘由。这类似于人类看到眼前的画面，但是眼中却没有注意焦点一样。

如果拿机器翻译来解释这个分心模型的Encoder-Decoder框架更好理解，比如输入的是英文句子：Tom chase Jerry，Encoder-Decoder框架逐步生成中文单词：“汤姆”，“追逐”，“杰瑞”。在翻译“杰瑞”这个中文单词的时候，分心模型里面的每个英文单词对于翻译目标单词“杰瑞”贡献是相同的，很明显这里不太合理，显然“Jerry”对于翻译成“杰瑞”更重要，但是分心模型是无法体现这一点的，这就是为何说它没有引入注意力的原因。

没有引入注意力的模型在输入句子比较短的时候问题不大，但是如果输入句子比较长，此时所有语义完全通过一个中间语义向量来表示，单词自身的信息已经消失，可想而知会丢失很多细节信息，这也是为何要引入注意力模型的重要原因。

上面的例子中，如果引入Attention模型的话，应该在翻译“杰瑞”的时候，体现出英文单词对于翻译当前中文单词不同的影响程度，比如给出类似下面一个概率分布值：


$$
(Tom,0.3)(Chase,0.2) (Jerry,0.5)
$$


每个英文单词的概率代表了翻译当前单词“杰瑞”时，注意力分配模型分配给不同英文单词的注意力大小。这对于正确翻译目标语单词肯定是有帮助的，因为引入了新的信息。

同理，目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。这意味着在生成每个单词的时候，原先都是相同的中间语义表示C会被替换成根据当前生成单词而不断变化的$C_i$。**理解Attention模型的关键就是这里，即由固定的中间语义表示C换成了根据当前输出单词来调整成加入注意力模型的变化的$C_i$。**增加了注意力模型的Encoder-Decoder框架理解起来如图2所示:

![image-20220515141236033](https://gitee.com/Rien_190/img-url/raw/master/img/202205151821033.png)

<center><p> 图2 引入注意力模型的Encoder-Decoder框架</p></center>

即生成目标句子单词的过程成了下面的形式：


$$
\begin{align}
y_1 = &f_1(C_1)\\
y_2 = &f_1(C_2,y_1)\\
y_3 = &f_1(C_3,y_1,y_2)
\end{align}
$$


而每个$C_i$可能对应着不同的源语句子单词的注意力分配概率分布，比如对于上面的英汉翻译来说，其对应的信息可能如下：


$$
\begin{align}
C_{汤姆} = &g(0.6*f_2("Tom"),0.2*f_2("chase"),0.2*f_2("Jerry"))\\
C_{追逐} = &g(0.2*f_2("Tom"),0.7*f_2("chase"),0.1*f_2("Jerry"))\\
C_{杰瑞} = &g(0.3*f_2("Tom"),0.2*f_2("chase"),0.5*f_2("Jerry"))\\
\end{align}
$$


其中，$f_2$函数代表Encoder对输入英文单词的某种变换函数，比如如果Encoder是用的RNN模型的话，这个$f_2$函数的结果往往是某个时刻输入$x_i$后隐层节点的状态值；g代表Encoder根据单词的中间表示合成整个句子中间语义表示的变换函数，一般的做法中，g函数就是对构成元素加权求和，即下列公式：


$$
C_i = \sum^{L_x}_{j=1}a_{ij}h_j
$$

其中，$L_x$代表输入句子Source的长度，$a_{ij}$代表在Target输出第i个单词时Source输入句子中第j个单词的注意力分配系数，而$h_j$则是Source输入句子中第j个单词的语义编码。假设$C_i$下标i就是上面例子所说的“ 汤姆” ，那么$L_x$就是3，h1=f(“Tom”)，h2=f(“Chase”),h3=f(“Jerry”)分别是输入句子每个单词的语义编码，对应的注意力模型权值则分别是0.6,0.2,0.2，所以g函数本质上就是个加权求和函数。如果形象表示的话，翻译中文单词“汤姆”的时候，数学公式对应的中间语义表示$C_i$的形成过程类似图3:

<img src="https://gitee.com/Rien_190/img-url/raw/master/img/202205151822347.png" alt="image-20220515173941242" style="zoom: 67%;" />

<center><p>图3 Attention的形成过程</p></center>

这里还有一个问题：生成目标句子某个单词，比如“汤姆”的时候，如何知道Attention模型所需要的输入句子单词注意力分配概率分布值呢？就是说“汤姆”对应的输入句子Source中各个单词的概率分布：(Tom,0.6)(Chase,0.2) (Jerry,0.2) 是如何得到的呢？

为了便于说明，我们假设对图2的非Attention模型的Encoder-Decoder框架进行细化，Encoder采用RNN模型，Decoder也采用RNN模型，这是比较常见的一种模型配置，则图1的框架转换为图4。

![image-20220515182524683](https://gitee.com/Rien_190/img-url/raw/master/img/202205151825368.png)

<center><p>图4 RNN作为具体模型的Encoder-Decoder框架</p></center>

那么用图5可以较为便捷地说明注意力分配概率分布值的通用计算过程。

![image-20220515193144507](https://gitee.com/Rien_190/img-url/raw/master/img/202205151931716.png)

<center><p>图5 注意力分配概率计算</p></center>

对于采用RNN的Decoder来说，在时刻i，如果要生成$y_i$单词，我们是可以知道Target在生成$y_i$之前的时刻i-1时，隐层节点i-1时刻的输出值$H_{i-1}$的，而我们的目的是要计算生成$y_i$时输入句子中的单词“Tom”、“Chase”、“Jerry”对$y_i$来说的注意力分配概率分布，那么可以用Target输出句子i-1时刻的隐层节点状态$H_{i-1}$去一一和输入句子Source中每个单词对应的RNN隐层节点状态$h_j$进行对比，即通过函数$F(h_j,H_{i-1})$来获得目标单词和每个输入单词对应的对齐可能性，这个F函数在不同论文里可能会采取不同的方法，然后函数F的输出经过Softmax进行归一化就得到了符合概率分布取值区间的注意力分配概率分布数值。

绝大多数Attention模型都是采取上述的计算框架来计算注意力分配概率分布信息，区别只是在F的定义上可能有所不同。

上述内容就是经典的Soft Attention模型的基本思想，那么怎么理解Attention模型的物理含义呢？一般在自然语言处理应用里会把Attention模型看作是输出Target句子中某个单词和输入Source句子每个单词的对齐模型，这是非常有道理的。

目标句子生成的每个单词对应输入句子单词的概率分布可以理解为输入句子单词和这个目标生成单词的对齐概率，这在机器翻译语境下是非常直观的：传统的统计机器翻译一般在做的过程中会专门有一个短语对齐的步骤，而注意力模型其实起的是相同的作用。

## Attention机制的本质思想

从图6来看待Attention机制:

![image-20220515200505865](https://gitee.com/Rien_190/img-url/raw/master/img/202205152005816.png)

<center><p>图6 Attention机制的本质思想</p></center>

将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。即可以将其本质思想改写为如下公式：


$$
Attention(Query,Source)=\sum^{L_x}_{i=1}Similarity(Query,Key_i)*Value_i
$$


其中，$L_x$代表Source的长度，公式含义即如上所述。上文所举的机器翻译的例子里，因为在计算Attention的过程中，Source中的Key和Value合二为一，指向的是同一个东西，也即输入句子中每个单词对应的语义编码，所以可能不容易看出这种能够体现本质思想的结构。

当然，从概念上理解，把Attention仍然理解为从大量信息中有选择地筛选出少量重要信息并聚焦到这些重要信息上，忽略大多不重要的信息，这种思路仍然成立。聚焦的过程体现在权重系数的计算上，权重越大越聚焦于其对应的Value值上，即权重代表了信息的重要性，而Value是其对应的信息。

 至于Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程

1. 阶段一: 根据Query和Key计算权重系数
   1. 根据Query和Key计算两者的相似性或者相关性
   2. 对1的原始分值进行归一化处理

2. 阶段二:根据权重系数对Value进行加权求和

这样，可以将Attention的计算过程抽象为如图7展示的三个阶段:

![image-20220515200858399](https://gitee.com/Rien_190/img-url/raw/master/img/202205152008398.png)

<center><p>图7 三阶段计算Attention过程</p></center>

在第一个阶段，可以引入不同的函数和计算机制，根据Query和某个$Key_i$，计算两者的相似性或者相关性，最常见的方法包括：求两者的向量点积、求两者的向量Cosine相似性或者通过再引入额外的神经网络来求值，即如下方式：


$$
\begin{align}
点积:&Similarity(Query,Key_i)=Query \cdot Key_i\\
Cosine相似性:&Similarity(Query,Key_i)=\frac{Query\cdot Key_i}{\lVert Query \rVert \cdot \lVert Key_i \rVert}\\
MLP网络:&Similarity(Query,Key_i)=MLP(Query,Key_i)
\end{align}
$$


第一阶段产生的分值根据具体产生的方法不同其数值取值范围也不一样，第二阶段引入类似SoftMax的计算方式对第一阶段的得分进行数值转换，一方面可以进行归一化，将原始计算分值整理成所有元素权重之和为1的概率分布；另一方面也可以通过SoftMax的内在机制更加突出重要元素的权重。即一般采用如下公式计算：


$$
a_i=Softmax(Sim_i)=\frac{e^{Sim_i}}{\sum^{L_x}_{j=1}e^{Sim_j}}
$$




第二阶段的计算结果$a_i$即为对$Value_i$应的权重系数，然后进行加权求和即可得到Attention数值：


$$
Attention(Query,Source)=\sum^{L_x}_{i=1}a_i \cdot Value_i
$$


通过如上三个阶段的计算，即可求出针对Query的Attention数值，目前绝大多数具体的注意力机制计算方法都符合上述的三阶段抽象计算过程。

## Self Attention模型

通过上述对Attention本质思想的梳理，我们可以更容易理解本节介绍的Self Attention模型。Self Attention也经常被称为intra Attention（内部Attention），最近一年也获得了比较广泛的使用。

在一般任务的Encoder-Decoder框架中，输入Source和输出Target内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target是对应的翻译出的中文句子，Attention机制发生在Target的元素Query和Source中的所有元素之间。而Self Attention顾名思义，指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的注意力计算机制。其具体计算过程是一样的，只是计算对象发生了变化而已，所以此处不再赘述其计算过程细节。

如果是常规的Target不等于Source情形下的注意力计算，其物理含义正如上文所讲，比如对于机器翻译来说，本质上是目标语单词和源语单词之间的一种单词对齐机制。那么如果是Self Attention机制，一个很自然的问题是：通过Self Attention到底学到了哪些规律或者抽取出了哪些特征呢？或者说引入Self Attention有什么增益或者好处呢？我们仍然以机器翻译中的Self Attention来说明.图8和图9是可视化地表示Self Attention在同一个英语句子内单词间产生的联系:

<img src="https://gitee.com/Rien_190/img-url/raw/master/img/202205152144184.png" alt="Snipaste_2022-05-15_21-43-01" style="zoom:60%;" />

<center><p>图8 可视化Self Attention实例</p></center>

![Snipaste_2022-05-15_21-46-53](https://gitee.com/Rien_190/img-url/raw/master/img/202205152147374.png)

<center><p>图9 可视化Self Attention实例</p></center>

从两张图可以看出，Self Attention可以捕获同一个句子中单词之间的一些句法特征（比如图8展示的有一定距离的短语结构）或者语义特征（比如图9展示的its的指代对象Law）。很明显，引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。

但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。除此外，Self Attention对于增加计算的并行性也有直接帮助作用。这是为何Self Attention逐渐被广泛使用的主要原因。



# Reference

[深度学习中的注意力机制(2017版)](https://blog.csdn.net/malefactor/article/details/78767781)
