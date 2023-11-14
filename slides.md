---
theme: ./theme
background: >-
  https://images.unsplash.com/photo-1487014679447-9f8336841d58?auto=format&fit=crop&q=80&w=2505&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## ai final
  final pre for kaggle AI competition shopee-product-matching
drawings:
  persist: false
title: final presentation
colorSchema: light
mdc: true
---

# AI 挑战项目结题汇报
基于多模态数据的 shopee 商品匹配


---

# 项目要求
## 


对每一个 post 求出一个 label_group.其中特征数据为：
* image 
* title


其中每一个 label_group 对应商品相同.


使用 $f1$ 分数评估，即精确率与召回率的调和平均数.

<img src="/image-13.png" style="height: 50%; margin: auto; display: block;" />

---

# 结题设计排名
## 


![res](/image.png)

目前 Private Score 分数 0.768， Public Score 分数 0.781.

在原比赛排名第5.


![best res](/image-1.png)

---

# baseline回顾
## 


在baseline中， 我们采用 TFIDF 提取文本特征， ResNet18 提取图片特征.


然后获取余弦相似值小于阈值的匹配，将文本得到的匹配和图像得到的匹配通过取并集的方式相接.

![baseline](/image-34.png)

---

# 训练模型
## 


baseline中未使用到给出的训练数据，预训练的参数不能很好地符合本次比赛数据.

我们初期使用 softmax loss 来训练图像和文本模型.

<br>

![softmax](/image-14.png)

---


# 开放集合目标检测
## 

根据其他参赛者的分析，我们得出此次比赛属于**开集检测**，即测试集的 label_group 大多没有出现在训练集中.


虽然大部分 label_group 不在训练集中， 但可以利用训练集增大不同大类（如衣服，食物，家具）之间的距离.



优点：稳定、表现好


缺点：耗时长

![triple-loss](/image-8.png)


---

# Arcface
## 

Triplet-loss 针对 **sample-to-sample** 距离的差异化，但本次比赛训练数据较大.


Arcface 则针对 **sample-to-class** 距离的差异化，有以下优点：
1. 能够有大量的图像比较（图与图，或者图与类）；
2. 有边界的概念；
3. 系统支持大规模训练


![arcface](/image-6.png)

---

# 图像处理模型 NFNet
High-Performance Large-Scale Image Recognition Without Normalization

* NFNet = Normalization Free Net.
* Batch Normalization： 防止梯度消失和梯度爆炸以及训练不稳定.但也增大计算成本.
* 自适应梯度裁剪模块：基于**逐单元梯度范数与参数范数的单位比例来裁剪梯度的算法**.

<br>

1. **梯度裁剪算法**：在更新 $\theta$ 之前，以如下公式裁剪梯度.（梯度向量 $G=\frac{\partial L}{\partial \theta}$,  $L$ 为损失值，$\theta$ 为模型所有参数向量，$\lambda$ 为需要调整的超参数）

$$

G \rightarrow
    \begin{cases}
    
    \lambda \frac{G}{\|G\|}& \text{if $\|G\| > \lambda$}, \\
    G & \text{otherwise.}
    \end{cases}

$$

> 一个超参数$\lambda$来控制所有层的梯度计算，需要改进.
---

# 图像处理模型 NFNet
## 

2. **AGC算法**：记 $W^l \in \mathbb{R}^{N*M}$ 为第 $l$ 层的权重矩阵, $G^l \in \mathbb{R}^{N*M}$为对应于$W^l$的梯度矩阵，$\| \cdot \|_F$ 表示 Frobenius 范数, 则有:


$$

\|W^\ell\|_F=\sqrt{\sum_{i=1}^N\sum_{j=1}^M(W_{i,j}^\ell)^2}

$$

<br>

定义第 $l$ 层上第 $i$ 个单元的梯度矩阵为$G_i^l$  （表示 $G^l$ 的第 $i$ 行），$\lambda$ 是一个超参数，<br> $\| W_i \|_F^*=\max (\| W_i \|_F, \epsilon)$ ， $\epsilon$ 默认为 $10^{-3}$ ，AGC算法的裁剪公式为：

$$G_i^\ell\to\begin{cases}\lambda\frac{\|W_i^\ell\|_F^\star}{\|G_i^\ell\|_F}G_i^\ell,&\mathrm{if~}\frac{\|G_i^\ell\|_F}{\|W_i^\ell\|_F^\star}>\lambda,\\G_i^\ell,&\mathrm{otherwise}.\end{cases}$$

---

# 文本处理模型 Sentence-BERT
##

若用 BERT 获取一个句子的向量表示，一般有两种方式：

1. 用句子开头的 [CLS] 经过 BERT 的向量作为句子的语义信息（这种更常用）.
2. 用句子中的每个 token 经过 BERT 的向量，加和后取平均，作为句子的语义信息.

<br> 

然而，实验结果显示，在文本相似度任务上，使用上述两种方法得到的效果并不好.

<img src="/image-37.png" style="height:50%; margin: auto; display: block;" />


---

# 文本处理模型 Sentence-BERT
## 

SBERT对预训练的BERT进行修改：

在 BERT 的输出结果上增加了一个 Pooling 操作，从而生成一个固定维度的句子 embedding.

使用（Siamese）和三级（triplet）网络结构来获得语义上有意义的句子嵌入，以此获得定长的句子嵌入.


对于分类问题有以下流程.

<img src="/image-36.png" style="height:50%; margin: auto; display: block;" />

---

# 文本处理模型 Sentence-BERT
## 

对于回归问题：计算两个句子 $u,v$ embedding 向量的余弦相似度（取值范围在 [-1,1] 之间）

<img src="/image-38.png" style="height:70%; margin: auto; display: block;" />

---

# 文本处理模型 Sentence-BERT
## 

为什么要将向量 $u,v,|u-v|$
 拼接在一起？文章针对不同的 concatenation 方法做了一系列实验，结果显示这种方法效果最好（同时结果也表明：Pooling 策略影响较小，向量组合策略影响较大）

<img src="/image-39.png" style="height:80%; margin: auto; display: block;" />


---

# 融合策略
## 

单模型的局限： **同一商品因为拍摄角度或者 title 描述方向的原因可能导致失配.**

基于此，我们采取**对两个模型得出的匹配结果取并集**的融合策略.

对多个模型取并集也是一种正则化的形式（类似集成学习中的 bagging）.

<img src="/image-12.png" style="height: 65%; margin: auto; display: block;" />


---

# Min2
## 

根据数据集的特性, 每个 label_group 的大小至少为 2 至多为 50.


采用 Min2 原则：我们令最近的点被匹配，即使其未达到阈值 threshold.即强制每组大小至少为2.



但如果最近点距离依旧超过min2_threshold， 说明该数据可能较为特殊，为保证精确率放弃此次匹配.


<br>

~~~python
idx = np.where(distances[k,] < threshold)[0] # 距离小于阈值为邻居，包括 k 自己

ids = indices[k,idx] # 找到 k 的邻居，包括 k 自己

if len(ids) <= 1 and distances[k,1] < min2_threshold: # 没有邻居，找最近一个
    ids = np.append(ids,indices[k,1])

~~~

<br>

由于大小为2的组较多，因此min2算法可以较为明显提高 F1 分数.

本次项目 min2_threshold 取0.6较为合适.

---


# Neighborhood Blendingneig
## 


使用近邻搜索和Min2阈值化得到图片的(matches, similarities)对，构造图，未通过阈值和Min2条件的节点不连接.

利用邻域信息细化查询图片的嵌入向量，通过相似性加权求取邻域嵌入向量的和，添加到查询嵌入向量中，称为NB（Neighborhood Blending）, 该方法可以更有效地聚合同类点.

<img src="/nbres.png" style="height: 65%; margin: auto; display: block;" />


---

# Neighborhood Blending
## 

<br>

<img src="/image-27.png"  />
---

# 单位提取
## 

在数据集中，我们发现部分数据图像一致，文本相似，但因为**单位**不同而导致失配.



我们通过**正则表达式**提取了所有的weight, length, pieces, memory, volume 单位.


若两个 title 在相同度量制下数值不同，则失配.

<br>

<img src="/image-26.png" style="height:50%; margin: auto; display: block;" />

---

# 融合策略的改进

在融合两个模型的时候，直接从匹配取并集的一个"缺点": 需要先通过 Threshold 得到匹配.


我们考虑先融合余弦相似性， 再从相似性直接得到匹配.


令 $a:= \text{similarity}_{\text{image}}<x,y>, \ \ \ b:= \text{similarity}_{\text{text}}<x,y>$



在以下四种方案中，实践证明第四种融合相似度的方法效果最好:

<br>

<img src="/image-21.png" style=" margin: auto; display: block;" />


---
layout: two-cols-header
---

# 概率意义下的并集

<img src="/image-21.png" style=" margin: auto; display: block;" />


::left::

事实上，余弦相似性可以看作两者匹配的概率:
* $a \in[0,1], \ \ \ b \in[0,1]$
* 余弦相似性越大，两者匹配概率越大

而 $a+b-a*b$ 即是概率意义下的并集.

::right::

<img src="/image-31.png" style="height:66%; margin: auto; display: block;" />

---

# chisel “凿子”
根据训练集的分布调整阈值 Threshold


我们提出一个假设:

> 比赛的训练集和测试集的目标 label_group 长度分布相同


<img src="/image-17.png" style="height:60%; margin: auto; display: block;" />


这可能与训练集和测试集划分有关系.


---

# chisel “凿子”
让测试集变成训练集的“形状”

![chisel](/image-35.png)
---

# 结题设计结构
## 

<br>

![final](/image-30.png)

---

# 结果分析

<img src="/image-40.png" style="height:60%; margin: auto; display: block;" />

<br>

* 本次比赛使用 eca-NFNet 以及 sentence-transformers/paraphrase-xlm-r-multilingual-v1.
* 本次比赛文本特征重要性略高于图像模型.

---

# 总结体会
## 

* 模型的提升和调参并没有带来太大的提升.
> Resnet18+Tfidf ➡️ NfNet-l1+SBERT(with Arcface),  0.762 ➡️ 0.768
* 后处理在本次比赛较为重要.
> 使用 chisel 后，  0.739 ➡️ 0.759 <br>
> 使用 Neighborhood Blendingneig 后，由银牌区➡️前10名
* 使用决策树也是另一种优秀的融合策略.
> 前几名部分的开源方案用了 LightGBM, XGBoost 等方法替代并集融合.
* 数据集较为嘈杂是限制 f1 分数的一个重要因素.
> "...noisy data is expected as that is the real situation in E-commerce since sellers have their own method of marketing, naming and promoting their products."
> ![Alt text](/image-33.png)

---

# 致谢
## 

<br>
<br>
<br>

* 感谢文泉老师细心指导
* 感谢 Kaggle 比赛讨论区各位大神 以及 github 用户jingxuanyang 的比赛总结
* 感谢 ChatGPT, Github Copilot 等AI工具的帮助

---
layout: cover
class: text-center
background: https://plus.unsplash.com/premium_photo-1661421746164-b8b53de3bd4e?q=80&w=2370&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
---

# 感谢聆听
#### 2023.11 第五小组

