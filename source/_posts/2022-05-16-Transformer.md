---
title: Transformer
date: 2022/5/16
categories: 科研
tags: [NLP, Transformer]
---

<meta name="referrer" content="no-referrer" />

NLP领域Transformer模型。

<!--more-->



Transformer是seq2seq model with self-attention.

# seq2seq

![image-20220516191914583](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205161919572.png)

seq2seq有很多应用, 这里不一一列举.

一般seq2seq model 有两个模块: encoder和decoder:

![image-20220516194154540](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205161941556.png)

# encoder

encoder作用:给一排向量, 输出另外一排向量,如图:

![image-20220516194317975](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205161943074.png)

transformer里用的是self-attention, transformer的encoder如图:

![image-20220516194528015](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205161945590.png)

每个block里有多个layer

上图右边是简化版本, 事实上每个block做的操作为:

![image-20220516213532551](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162135444.png)

图中需要做两点说明:

1. 计算$x_i^{'}$时等号右边为$x_i$
2. 左边norm的输出是右边FC的输入

右边norm的输出就是一个block的输出.

# decoder

decoder作用为产生输出.

## AT

decoder将encoder的输出先读入, 给decoder一个特殊符号表示开始BOS(begin of sentence), decoder读到特殊符号后会输出一个向量, 该向量和vocabulary的长度相同. 在产生输出向量之前都会做softmax, 该向量会给每一个元素一个值, 值最大的就是输出. 如图所示:

![image-20220516215338595](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162153701.png)

将这个输出连同BOS当作输入, 再得到下一个输出:

![image-20220516215408696](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162154606.png)

如此重复:

![image-20220516215557890](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162155365.png)

对比encoder和decoder内部结构:

![image-20220516215655970](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162156026.png)

如果把decoder的中间遮住:

![image-20220516215720200](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162157067.png)

encoder和decoder是一样的. 只是decoder最后会做一个softmax得到几率

不一样之处在于decoder还有一个mask. masked self-attention和self-attention有一定区别, 下面列举出的图片可以和[Self-Attention机制](https://rien190.github.io/2022/05/15/self-attentionn.html#self-attention%E6%A8%A1%E5%9E%8B)中图片进行对比理解

![image-20220516220016462](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162200110.png)

每次产生的vector只能看到前面的vector, eg $b^2$只能看到$a^1和a^2$.

更具体来说:

![image-20220516220535863](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162205736.png)

因为在self-attention中$a^1到a^4$是同时出现的, 而在decoder中$a^1到a^4$是顺序出现的.

还有问题需要处理: decoder需要决定输出的长度. 所以需要END符. 如图:

![image-20220516221311870](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162213307.png)



## NAT vs AT

![image-20220516222536334](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162225367.png)

## Cross attention

![image-20220516222735582](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162227503.png)

实际运作过程:

![image-20220516222918178](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162229991.png)

# Training

![image-20220516224154992](https://raw.githubusercontent.com/Rien190/ImgURL/master/img/202205162241023.png)

训练时需要缩小decoder的输出与正确答案之间的差距,训练时decoder的输入为正确答案





# reference

[Transformer 李宏毅](https://www.youtube.com/watch?v=ugWDIIOHtPA)

[Self-Attention机制](https://rien190.github.io/2022/05/15/self-attentionn.html#self-attention%E6%A8%A1%E5%9E%8B)