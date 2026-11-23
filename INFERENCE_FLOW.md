# OV-Stitcher 推理链路与张量形状

本文按一次 VOC21 单图评测说明数据如何从输入像素变成最终分割图。除非特别说明，形状顺序遵循 PyTorch：图像为 `NCHW`，CLIP/DINO token 为 `NLC` 或 `LNC`。`H/W` 是 MMSeg 预处理后的尺寸，`H0/W0` 是原图尺寸。

## 1. 数据加载与预处理

`evaluation.py` 将配置交给 MMEngine Runner。MMSeg 测试 pipeline 读取图像、缩放并读取标注；`SegDataPreProcessor` 将 BGR 转为 RGB，并按 CLIP 的均值与标准差归一化。

- 单张图像：`[1, 3, H, W]`
- 配置中的区域掩码 `.npz`：`[H0, W0]` 整数 ID 图
- 原始掩码堆叠后：`[1, H0, W0]`
- 掩码最近邻缩放到模型输入后：`[1, 1, H, W]`

掩码使用最近邻插值，因为区域 ID 是离散整数；双线性插值会生成无效 ID。原始分辨率掩码另行保留，用于最后的区域多数投票。

## 2. 重叠裁剪

`make_sliding_windows` 按配置的 `slide_crop` 和 `slide_stride` 切分归一化图像。重叠区域让边界 patch 也能被窗口完整覆盖；`coords` 记录每个窗口在整图中的像素坐标，`paddings` 记录为 patch 对齐增加的填充。

- 输入：`[1, 3, H, W]`
- 窗口图像：`[Nc, 3, Hc, Wc]`
- 坐标：`Nc` 个 `(y1, x1, y2, x2)`
- 填充：`Nc` 个 `(left, right, top, bottom)`

常用 `slide_crop` 为 224 或 336。尺寸不足时会补到 56 的倍数，使 DINO patch 网格在窗口间对齐。后续合并会依据坐标把 crop token 放回全图网格，并对重叠处取平均。

## 3. DINO：把窗口末层注意力改成整图注意力

`stitch_dino_features` 先撤销输入上的 CLIP 归一化，再应用 ImageNet 归一化，因为 DINO 使用自己的输入统计量。`dino_vitb8` 对每个 crop 产生 patch size 为 8、通道数为 768 的 token。

1. DINO 对所有窗口做前向，得到末两层特征；hook 读取最后一个 block 的投影结果 `[Nc, Lcrop+1, 3*768]`。
2. 去掉 CLS token，因为它没有像素坐标。拆分后的每个 crop Q、K、V 和倒数第二层 patch 特征都是 `[Nc, Lcrop, 768]`。
3. 根据 `coords/paddings` 将各窗口 token 放回全图 patch 网格；重叠位置取平均，得到 `[1, Nd, 768]`，其中 `Nd=Hd*Wd`、`Hd=H//8`、`Wd=W//8`。
4. 将 Q/K/V 拆成 attention heads，计算整图的 `QKᵀ / sqrt(head_dim)`，形状为 `[heads, Hd*Wd, Hd*Wd]`。这让不同 crop 的 patch 在最后一个 DINO block 中交换上下文。
5. attention 输出经过原 DINO block 的输出投影，与 stitched 倒数第二层特征做残差相加；再执行该 block 的 LayerNorm、MLP 和第二次残差相加。
6. 对最终 token 做 L2 归一化，输出 `dino_features=[1, Hd*Wd, 768]`。

原始 DINO 前向仍按 crop 执行所有 block，用来取得倒数第二层特征和最后层 Q/K/V；逐 crop 得到的最后层输出不作为结果，而是在 stitched 网格上按同一组权重重算。自注意力矩阵随全图 token 数平方增长，因此输入分辨率会影响显存。

## 4. 区域范围与 DINO 相似度

CLIP 适配器把输入掩码最近邻缩放到 DINO 网格，得到 `[1, Hd, Wd]`。DINO token 两两点积得到 `[1, Hd*Wd, Hd*Wd]` 相似度矩阵。

- 同一区域内的 patch 保留互相注意的资格。
- 前景区域可按区域平均 DINO 特征做 DBSCAN 合并；新 ID 使用独立编号，避免与原区域 ID 冲突。
- 背景仍可与 DINO 相似度高于全局均值的 patch 建立连接，减少背景被完全隔离的情况。
- 被禁止的 pair 设为 `-inf`，乘温度系数 4 后沿 key 维做 softmax，形成 `[1, Hd*Wd, Hd*Wd]` 的范围注意力。

这个矩阵在 CLIP 最后一层融合时提供 DINO 引导的全局权重；区域掩码负责约束连接范围，特征相似度负责背景连接。

## 5. CLIP：窗口、空间和语义三条特征流

项目使用公开 `open_clip_torch==2.24.0` 提供的 `ViT-B/16`、文本编码器、权重和 tokenizer。该版本与作者原仓库内标注的 OpenCLIP 版本一致；可对照上游 [model.py](https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/model.py) 和 [transformer.py](https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/transformer.py)。`models/clip_stitching.py` 只负责作者增加的高分辨率视觉路径，不修改 OpenCLIP 源码。

### 5.1 窗口 token

对每个 CLIP crop 执行 patch convolution：

- 输入：`[Nc, 3, Hc, Wc]`
- patch map：`[Nc, 768, Hc/16, Wc/16]`
- patch token：`[Nc, Lclip, 768]`，其中 `Lclip=(Hc/16)*(Wc/16)`
- 加入一个 CLS token 和位置编码：`[Nc, Lclip+1, 768]`
- LayerNorm 后转为 OpenCLIP block 使用的 `LNC`：`[Lclip+1, Nc, 768]`

位置编码来自预训练的 14×14 网格。crop 大于 224×224 时用 bicubic 插值扩展到 crop patch 网格，避免位置编码长度与 token 数不匹配。

### 5.2 区域语义 token 和局部注意力

把合并后的 DINO 区域 ID 上采样回输入像素，再按相同窗口裁剪并缩放到 CLIP patch 网格。每个区域复制一个 CLS 表征，作为区域查询：

- 区域掩码：`[Nc, Nr, Hclip, Wclip]`
- 区域查询：`[Nr, Nc, 768]`
- 与图像 token 拼接后：`[Nr+1+Lclip, Nc, 768]`
- 每个 crop 的 bool attention mask：`[Nc, Nr+1+Lclip, Nr+1+Lclip]`

上游 OpenCLIP v2.24 的 residual block 会先把 bool mask 转成 token 浮点类型，再交给 `MultiheadAttention`（见 [attention mask 转换](https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/transformer.py#L197-L210)）。因此 bool 值实际成为加到注意力 logits 上的 0/1 bias，而不是硬性屏蔽；本次重构保留这一原始行为，没有改成 `-inf` 屏蔽，以免悄悄改变作者算法。这个 bias 调整不同区域配对的注意力权重。

### 5.3 中间层空间分支

每个选定 CLIP block 同时做两件事：

- 窗口分支：各 crop 按作者保存的区域 mask bias 经过该 block，保持 `[L, Nc, 768]`。
- 整图分支：取窗口 patch token 和该 block 的 Q/K/V，按像素坐标 stitch 到全图 CLIP 网格；在全图 patch 序列上执行 attention 和 MLP，输出 `[1, Hcgrid*Wcgrid, 768]`。

ViT-B/16 有 12 个 block，空间分支采用第 5 到第 9 号 block（从 0 开始计数）的输出并取平均。局部窗口流保留对象细节，整图流补入跨窗口空间上下文。

### 5.4 最后一层融合

最后一个 CLIP block 生成三路对齐到 DINO 网格的特征：

1. **main**：窗口 patch 的 V 投影在 CLIP 网格 stitch 后，插值到 `[Hd, Wd]`。
2. **spatial**：中间层整图特征经过最后 block 的 V 投影，再插值到 `[Hd, Wd]`。
3. **semantic**：区域查询经过最后 block 后，按每个区域的 patch membership 汇聚成窗口内语义 token，再 stitch 到 DINO 网格。

main 和 spatial 两路拼成 `[2, Hd*Wd, 768]`，由 DINO 范围注意力分别聚合；semantic 为 `[1, Hd*Wd, 768]`。三路沿分支维拼接为 `[3, Hd*Wd, 768]`，应用 CLIP `ln_post` 后按 `main + spatial + 0.5*semantic` 融合，最后乘公开 OpenCLIP 的视觉投影矩阵 `[768,512]`，得到 `[1, Hd*Wd, 512]`。

这里保留作者实现的 QK 计算方式、层选择和融合系数；适配层把每一步拆成具名函数，并明确窗口布局、mask 方向和 token 网格。

## 6. 文本类别原型

类别文件每一行是一个类别；`; ` 分隔的名称会作为该类别的别名。对每个别名，把通用 ImageNet 模板和可选类别模板变成文本：

- tokenizer 输出：`[T, 77]`
- CLIP 文本向量：`[T, 512]`
- 每条向量先归一化，再对 T 个模板取平均并再次归一化：`[512]`
- 所有别名堆叠：`query_features=[Nalias,512]`

预先编码文本原型只做一次，避免每张图重复运行文本 Transformer。

## 7. 像素分类和区域校正

归一化图像 token 与文本原型点积：

- `image_features`: `[1, Hd*Wd, 512]`
- `query_features.T`: `[512, Nalias]`
- logits: `[1, Hd*Wd, Nalias]`
- reshape 后：`[1, Nalias, Hd, Wd]`

logits 先双线性插值到预处理尺寸 `[H,W]`，再按 MMSeg 元信息插值到 `[H0,W0]`。按类维乘 logit scale 40 并 softmax；同一类别的背景别名用 max 合并。低于 `prob_thd` 的像素设为背景 ID 0。

最后读取原分辨率区域图 `[H0,W0]`，将每个非零区域的像素统一为该区域出现次数最多的类别，输出 `[1,H0,W0]` 预测标签并交给 MMSeg IoU evaluator。

## 8. 关键实现位置

- 入口与 Runner：`evaluation.py`
- 数据集及类别配置：`configs/datasets/`、`configs/classes/`
- 窗口、DINO Q/K/V stitching：`models/features.py`
- 公共 OpenCLIP 的作者定制视觉路径：`models/clip_stitching.py`
- MMSeg 模型与输出恢复：`models/segmentor.py`
- 文本原型：`models/text.py`
- 区域加载和多数投票：`models/masks.py`
