# Spiral RoPE 6: Rotate Your Rotary Positional Embeddings in the 2D Plane

Haoyu Liu<sup>1</sup> Sucheng Ren<sup>2</sup> Tingyu Zhu<sup>1</sup> Peng Wang<sup>3</sup> Cihang Xie<sup>4</sup> Alan Yuille<sup>2</sup> Zeyu Zheng<sup>1</sup> Feng Wang<sup>2</sup>

# **Abstract**

Rotary Position Embedding (RoPE) is the de facto positional encoding in large language models due to its ability to encode relative positions and support length extrapolation. When adapted to vision transformers, the standard axial formulation decomposes two-dimensional (2D) spatial positions into horizontal and vertical components, implicitly restricting positional encoding to axis-aligned directions. We identify this directional constraint as a fundamental limitation of the standard axial 2D RoPE, which hinders the modeling of oblique spatial relationships that naturally exist in natural images. To lift this limitation, we propose **Spiral RoPE**, a simple yet effective extension that enables multi-directional positional encoding by partitioning embedding channels into multiple groups associated with uniformly distributed directions. Each group is rotated according to the projection of the patch position onto its corresponding direction, allowing spatial relationships to be encoded beyond the horizontal and vertical axes. Across a wide range of vision tasks including classification, segmentation, and generation, Spiral RoPE consistently improves performance. Qualitative analysis of attention maps further show that Spiral RoPE exhibits more concentrated activations on semantically relevant objects and better respects local object boundaries, highlighting the importance of multi-directional positional encoding in vision transformers.

## 1. Introduction

Rotary Position Embedding (RoPE) (Su et al., 2023) has become a cornerstone of modern language transformers, powering state-of-the-art large language models such as the

Code and model checkpoints available at: https://github.com/huajianduzhuo-code/Spiral\_RoPE <sup>1</sup>University of California, Berkeley <sup>2</sup>John Hopkins University <sup>3</sup>ByteDance <sup>4</sup>University of California, Santa Cruz. Correspondence to: Feng Wang <wangf3014@gmail.com>.

Preprint. February 4, 2026.

![](_page_0_Figure_9.jpeg)

![](_page_0_Figure_10.jpeg)

Figure 1. Frequency support visualization comparing Axial 2D RoPE (left) and Spiral RoPE (right) with eight rotation groups (K=8). As shown, the Axial 2D RoPE places all frequencies on horizontal and vertical axes only, while our Spiral RoPE distributes frequencies across multiple directions in a spiral pattern, offering a broader directional coverage.

LLaMA series (Touvron et al., 2023a;b; Grattafiori et al., 2024) and Qwen series (Bai et al., 2023; Yang et al., 2024; 2025b;a). By encoding positional information through rotation operations on query and key vectors, RoPE naturally captures relative positions and enables effective length extrapolation. Such properties have been proven essential for scaling transformers to long contexts. In contrast, standard Vision Transformers (ViTs) (Dosovitskiy et al., 2021) predominantly rely on absolute positional embeddings (APE), which encode fixed position vectors that are added to patch embeddings. While simple and effective, APE has well-documented limitations: it struggles to generalize to resolutions unseen during training and provides no explicit mechanism for encoding relative spatial relationships between image patches.

Recent work has begun exploring RoPE for vision tasks (Fang et al., 2024; Lu et al., 2024; 2023), demonstrating promising results on image classification, dense prediction, and generation tasks. The standard approach extends 1D RoPE to 2D images by decomposing spatial positions into horizontal and vertical components. We term this approach as *Axial 2D RoPE*. Half of the embedding dimensions are rotated based on the x-coordinate, while the other half uses the y-coordinate. However, this axial decomposition inherits a fundamental limitation: it can only encode positional relationships along the coordinate axes. when we visualize the frequency support of Axial 2D RoPE in the 2D Fourier domain, all frequencies lie exclusively on the horizontal and vertical axes. This means Axial 2D RoPE is insensitive to positional changes along diagonal

![](_page_1_Figure_1.jpeg)

Figure 2. Fourier reconstruction comparisons. Given binary input images (*left*), we retain only frequencies representable by each RoPE method and reconstruct via inverse-FFT. Axial 2D RoPE (*middle*) produces artifacts along horizontal and vertical axes. Spiral RoPE achieves more faithful reconstruction due to broader directional coverage.

or other oblique directions, which is a limitation given that natural images contain rich spatial structures in all orientations. To demonstrate this limitation concretely, we perform a Fourier reconstruction experiment in Figure 2: given a binary image, we retain only the frequency components representable by each RoPE variant and reconstruct via inverse FFT (Fast Fourier Transform). Axial 2D RoPE produces visible artifacts along the coordinate axes, failing to faithfully reconstruct the input image structures such as circles.

In this work, we introduce **Spiral RoPE**, a new 2D rotary position embedding that extends directional coverage beyond the coordinate axes. Instead of splitting embeddings into just two axis-aligned groups, Spiral RoPE partitions them into K groups corresponding to K uniformly distributed directions in the 2D space. Each embedding group is rotated based on the patch position projected onto its assigned direction, enabling the model to explicitly encode positional relationships along diagonal and other oblique orientations. We design a grouped interleaved frequency assignment strategy that distributes the same number of distinct frequencies as Axial 2D RoPE across all K directions, ensuring no loss in multi-scale encoding capacity. As shown in Figure 1, the resulting frequency pattern forms a *spiral* in the 2D frequency plane, providing broader directional coverage.

Notably, Spiral RoPE is simple to implement and introduces no additional computational overhead compared to Axial 2D RoPE, as both methods apply the same number of rotation operations with the same frequency budget, and our method does not introduce any additional learnable parameters. However, this simple change yields surprisingly significant and consistent improvements across diverse vision tasks. For example, on the standard ImageNet-

![](_page_1_Figure_6.jpeg)

Figure 3. Query-token attention visualization. The query token (marked by a red box) is located on a foreground object. Across diverse scenes, Spiral RoPE consistently produces sharper and more localized attention around the queried region compared to APE and Axial RoPE, indicating improved spatial alignment.

1k (Deng et al., 2009) classification task, Spiral RoPE improves test accuracy by +0.7% for ViT-Large and +1.0% for ViT-Base *with minimal cost*. For semantic segmentation on ADE20k (Zhou et al., 2017), our Spiral RoPE consistently achieves +2.2% mIoU improvement for a ViT-Large backbone and +1.2% mIoU for a ViT-Base. On class-conditional image generation with Diffusion Transformers (DiT) (Peebles & Xie, 2023), Spiral RoPE reduces FID by 3.9–5.8 points across various model sizes. We also observe that Spiral RoPE demonstrates superior extrapolation capabilities to unseen resolutions, especially when the evaluation resolution is higher than the training resolution.

Beyond quantitative improvements, qualitative analyses further suggest that Spiral RoPE leads to more effective spatial representations. As illustrated in Figure 2, Spiral RoPE enables more faithful Fourier-domain reconstruction than Axial 2D RoPE, yielding reconstructions that more closely resemble the original images while avoiding characteristic axis-aligned artifacts. Consistent trends are also observed in attention map visualizations in Figure 3, where we visualize attention patterns from individual query tokens on the foregroound object. Spiral RoPE produces sharper and more spatially coherent attention concentrated around the query location and semantically related regions, in contrast to the diffuse patterns produced by APE and the axis-biased activations of Axial RoPE. Together, these observations indicate that the directional constraints inherent in Axial 2D RoPE can limit spatial modeling in two-dimensional settings. By relaxing this restriction through a simple geometric extension, Spiral RoPE provides a more flexible positional encoding that supports richer spatial interactions, leading to improved spatial understanding in vision transformers. We hope this work can inspire generalizable architectural improvements for Vision Transforms.

## 2. Related Work

Positional Encoding in Transformers. Transformers (Vaswani et al., 2023) lack inherent positional awareness due to the permutation-equivariant nature of self-attention. To address this, the original Transformer used sinusoidal absolute position embeddings (APE), which add fixed positiondependent signals to input tokens. Learnable APE (Devlin et al., 2019) replaces fixed sinusoids with trainable vectors, offering greater flexibility. However, APE methods encode absolute positions, which limits their ability to generalize to sequence lengths unseen during training. Relative position encodings address this limitation by encoding the distance between tokens rather than their absolute positions. Shaw et al. (Shaw et al., 2018) introduced learnable relative position biases added to attention logits. T5 (Raffel et al., 2023) further simplified this approach with a small number of learnable bias buckets. ALiBi (Press et al., 2022) proposed a non-learned linear bias that scales with distance, enabling length extrapolation in language models. When extending Transformers from 1D sequences to 2D visual data, vision Transformer (ViT) (Dosovitskiy et al., 2021) applies Transformers to images by dividing them into patches, using learnable APE. DeiT (Touvron et al., 2021) enabled efficient ViT training on ImageNet-1K, while Swin Transformer (Liu et al., 2021) introduced relative position bias within local windows. These developments highlight the importance of positional encoding design in vision models for handling varying resolutions and capturing spatial relationships.

Rotary Position Embedding. Rotary Position Embedding (RoPE) (Su et al., 2023) introduces a fundamentally different approach to position encoding by applying rotation operations directly to query and key vectors. Unlike additive position encodings, RoPE multiplies the query and key representations by rotation matrices whose angles are determined by the position index and a set of base frequencies. This design ensures that the dot product between query and key depends only on their relative positions, inherently encoding relative positional information. RoPE has become the de facto position encoding for modern large language models including LLaMA (Touvron et al., 2023a) and Qwen (Yang et al., 2025a), demonstrating excellent length extrapolation capabilities. Various extensions have been proposed to improve RoPE's ability to handle long sequences, including position interpolation (Chen et al., 2023) which rescales positional indices at inference time and YaRN (Peng et al., 2023) which refines this strategy through frequency-aware scaling to improve stability.

**2D RoPE for Vision.** Extending RoPE to 2D vision data requires careful design choices. The straightforward approach, which we term axial 2D RoPE, splits the embedding dimension into two halves and applies 1D RoPE independently along the x and y axes. The 2D RoPE method has been adopted in many vision tasks, such as SAM 2 (Ravi et al., 2024) for image segmentation and Qwen-Image (Wu et al., 2025) for image generation. M-RoPE (Multimodal RoPE) (Wang et al., 2024) decomposes image indices into height, width, and temporal components to align visual patches with 1D text sequences. However, these axial approaches encode only horizontal and vertical relationships, without explicit awareness of diagonal or other directional spatial patterns. Heo et al. (Heo et al., 2024) proposed RoPE-Mixed, treats the frequencies for both spatial axes as learnable parameters, allowing the network to adaptively "mix" axial signals to capture diagonal information. In multimodal settings, several works generalize RoPE to jointly handle tokens from different modalities. For instance, Circle-RoPE (Wang et al., 2025) project image token indices onto a ring that is orthogonal to the linear axis of text token indices to eliminate spurious cross-modal biases.

#### 3. Method

## 3.1. Preliminaries: Rotary Position Embedding

### 3.1.1. 1D RoPE

Rotary Position Embedding (Su et al., 2023) encodes positional information through rotation operations. Given a d-dimensional query or key vector  $\mathbf{x} \in \mathbb{R}^d$  at position m, RoPE applies a block-diagonal rotation matrix:

$$RoPE(\mathbf{x}, m) = \mathbf{R}_m \mathbf{x} \tag{1}$$

where  $\mathbf{R}_m$  is a block-diagonal matrix composed of d/2 2D rotation matrices:

$$\mathbf{R}_m = \begin{pmatrix} \mathbf{R}_m^{(0)} & & \\ & \ddots & \\ & & \mathbf{R}_m^{(d/2-1)} \end{pmatrix} \tag{2}$$

with each  $2 \times 2$  block being:

$$\mathbf{R}_{m}^{(t)} = \begin{pmatrix} \cos(m\theta_{t}) & -\sin(m\theta_{t}) \\ \sin(m\theta_{t}) & \cos(m\theta_{t}) \end{pmatrix}$$
(3)

The frequency  $\theta_t$  is typically set as  $\theta_t = \theta_{\rm base}^{-t/(d/2)}$  for  $t=0,1,\ldots,d/2-1$ , where  $\theta_{\rm base}$  is the base frequency that is commonly set to 10,000. This geometric progression of frequencies allows RoPE to encode positional dependencies at multiple scales.

A key property of RoPE is that the dot product between rotated vectors depends only on their relative position:

$$\langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle = \mathbf{q}^{\top} \mathbf{R}_{m-n} \mathbf{k}$$
 (4)

This property enables RoPE to implicitly encode relative positional information when calculating the attention matrix in the self-attention mechanism.

#### 3.1.2. AXIAL 2D ROPE

RoPE was originally designed for 1D sequences. For 2D images, the standard approach is to extend RoPE by decomposing the spatial position into x and y coordinates. Given a patch at position  $(p_x, p_y)$ , the d-dimensional embedding  $\mathbf{x} \in \mathbb{R}^d$  is split into two halves:

$$\mathbf{x}^{(x)} = (x_0, x_1, \dots, x_{d/2-1}) \tag{5}$$

$$\mathbf{x}^{(y)} = (x_{d/2}, x_{d/2+1}, \dots, x_{d-1})$$
 (6)

The first half is rotated using the x-coordinate, and the second half using the y-coordinate:

RoPE-2D(
$$\mathbf{x}, p_x, p_y$$
) =  $\begin{pmatrix} \mathbf{R}_{p_x} \mathbf{x}^{(x)} \\ \mathbf{R}_{p_y} \mathbf{x}^{(y)} \end{pmatrix}$  (7)

The rotation matrices  $\mathbf{R}_{p_x}$  and  $\mathbf{R}_{p_y}$  are typically constructed using the same set of frequencies. Each sub-embedding  $\mathbf{x}^{(x)}$  and  $\mathbf{x}^{(y)}$  has dimensionality d/2 and thus can only accommodate d/4 distinct rotary frequencies, i.e., half of those available in the original 1D RoPE.

Despite the good simplicity of transferring 1D RoPE to 2D inputs, this axial design only encodes positional relationships along the axis-aligned directions, leading to an inherent limitation. The x-coordinate and y-coordinate are processed independently, without any explicit encoding of diagonal or other directional relationships. Consider patches at positions  $\mathbf{p}_1 = (0,0)$  and  $\mathbf{p}_2 = (1,1)$ , which are diagonally adjacent. With conventional 2D RoPE, the relative position encoding in the x-dimensions reflects  $\Delta x = 1$ , and in the y-dimensions reflects  $\Delta y = 1$ , but there is no direct encoding of the diagonal relationship.

#### 3.2. Spiral RoPE

To address this limitation, we propose to extend 2D RoPE by applying additional *spatial rotations* on the 2D panel. For example, we can directly encode the diagonal displacement  $\sqrt{2}$  along the 45-degree direction, which can provide richer positional information. And more generally, we can encode positions along multiple uniformly distributed directions, which provides the model with a more complete description of spatial relationships.

Formally, let K be the number of directions we want to encode. We uniformly distribute K directions in the angular range  $[0, \pi)$ :

$$\phi_k = \frac{k\pi}{K}, \quad k = 0, 1, \dots, K - 1$$
 (8)

Note that we only need to cover  $[0,\pi)$  rather than  $[0,2\pi)$  because directions  $\phi$  and  $\phi+\pi$  are parallel, and RoPE naturally handles both positive and negative positions through its rotation mechanism.

For each direction  $\phi_k$ , we define a unit direction vector:

$$\mathbf{u}_k = (\cos \phi_k, \sin \phi_k) \tag{9}$$

Given a patch at 2D position  $\mathbf{p} = (p_x, p_y)$ , we compute its projected position along direction k as:

$$t_k(\mathbf{p}) = \mathbf{p} \cdot \mathbf{u}_k = p_x \cos \phi_k + p_y \sin \phi_k \tag{10}$$

We partition the d-dimensional embedding into K equal groups, each with d/K dimensions. The k-th group, denoted  $\mathbf{x}^{(k)}$ , is rotated using the projected position  $t_k$ :

Spiral RoPE(
$$\mathbf{x}, \mathbf{p}$$
) = 
$$\begin{pmatrix} \mathbf{R}_{t_0(\mathbf{p})}^{(0)} \mathbf{x}^{(0)} \\ \mathbf{R}_{t_1(\mathbf{p})}^{(1)} \mathbf{x}^{(1)} \\ \vdots \\ \mathbf{R}_{t_{K-1}(\mathbf{p})}^{(K-1)} \mathbf{x}^{(K-1)} \end{pmatrix}$$
(11)

where  $\mathbf{R}_t^{(k)}$  denotes the rotation matrix for direction k with its specific frequency assignment.

A critical design choice in Spiral RoPE is how to assign rotation frequencies  $\theta_i$  to different directions. A naive approach would independently assign d/(2K) frequencies to each direction. However, this limits the number of distinct frequencies as K increases, reducing the model's ability to encode positions at multiple scales.

To maximize frequency diversity, we adopt an interleaved frequency assignment strategy. We first compute a pool of d/4 base frequencies using the same formula as axial 2D RoPE:

$$\theta_t = \theta_{\text{base}}^{-t/(d/4)}, \quad t = 0, 1, \dots, d/4 - 1$$
 (12)

This gives us the same number of distinct frequencies as the conventional 2D RoPE baseline.

To distribute these frequencies across directions, we adopt a grouped interleaved assignment strategy. We first group adjacent frequencies into pairs:  $(\theta_0,\theta_1),(\theta_2,\theta_3),\ldots,(\theta_{d/4-2},\theta_{d/4-1})$ . We then assign these pairs to direction groups in a round-robin fashion. Specifically, we pair directions that are 90° apart (*i.e.*, perpendicular), since they encode orthogonal spatial relationships. For direction pair (k,k+K/2) where k < K/2, we assign frequency pairs:

$$\Theta^{(k)} = \Theta^{(k+K/2)} = \{\theta_{2k}, \theta_{2k+1}, \theta_{2k+K}, \theta_{2k+K+1}, \ldots\}$$
(13)

For example, with K=4 directions (0°, 45°, 90°, 135°) and d=32, the frequency assignment is:

- $\theta_0, \theta_1, \theta_4, \theta_5$  are assigned to directions  $0^{\circ}$  and  $90^{\circ}$
- $\theta_2, \theta_3, \theta_6, \theta_7$  are assigned to directions 45° and 135°

Figure 1 visualizes the frequency assignment by plotting each frequency  $\pm\theta_t$  as a point in the 2D frequency plane along the direction it encodes. For traditional 2D axial RoPE, all frequencies lie on the horizontal (0°) and vertical (90°) axes, corresponding to frequency vectors  $(\pm\theta_t,0)$  and  $(0,\pm\theta_t)$ . In contrast, Spiral RoPE with K=8 distributes frequencies across K/2=4 pairs of perpendicular directions:  $0^\circ/90^\circ, 22.5^\circ/112.5^\circ, 45^\circ/135^\circ,$  and  $67.5^\circ/157.5^\circ,$  which forms a spiral pattern. This visualization illustrates how Spiral RoPE expands directional coverage beyond the coordinate axes while maintaining the same total number of frequencies.

This grouped interleaved assignment ensures that: (1) we utilize all d/4 distinct frequencies, matching the capacity of axial 2D RoPE; (2) perpendicular direction pairs share the same frequency set, analogous to how axial 2D RoPE uses the same frequencies for both x and y axes; and (3) each direction receives a mix of low and high frequencies from adjacent pairs, enabling multi-scale positional encoding within each direction.

Spiral RoPE preserves the relative position encoding property of RoPE: for patches at positions  $\mathbf{p}$  and  $\mathbf{p}'$ , since  $t_k(\mathbf{p}) - t_k(\mathbf{p}') = t_k(\mathbf{p} - \mathbf{p}')$ , the attention score contribution depends only on the relative position  $\mathbf{p} - \mathbf{p}'$ . Furthermore, by uniformly distributing directional encodings, Spiral RoPE provides more isotropic spatial awareness compared to axial 2D RoPE, which treats horizontal and vertical directions preferentially. Note that Spiral RoPE can be implemented efficiently by precomputing the rotation matrices for all positions and directions, with minimal computational overhead compared to axial 2D RoPE.

To further illustrate the representation power of Spiral RoPE, we perform a 2D Fourier analysis comparing axial 2D RoPE with Spiral RoPE. We consider d = 1024, which yields d/4 = 256 distinct RoPE frequencies. Given a binary image, we first apply 2D FFT to obtain its frequency-domain representation. We then retain only the frequency components that can be represented by each RoPE variant: for axial 2D RoPE, we keep frequencies along the horizontal and vertical axes  $(\pm \theta_t, 0)$  and  $(0, \pm \theta_t)$ ; for Spiral RoPE, we keep frequencies along all K = 8 directions as illustrated in Figure 1. All other frequency components are masked to zero. Finally, we apply 2D iFFT to reconstruct the image and compare the result with the original input. Figure 2 shows the reconstruction results for two input images: a single point and a circle on the  $64 \times 64$  grid. The axial 2D RoPE reconstruction exhibits visible artifacts along the horizontal and vertical axes, as it can only represent frequency components in these two directions. In contrast, Spiral

RoPE produces reconstructions that are closer to the original images, demonstrating its ability to capture a broader range of spatial patterns through multi-directional frequency coverage.

# 4. Experiments

We conduct comprehensive experiments to validate the effectiveness of Spiral RoPE across multiple vision tasks: image classification on ImageNet-1k, semantic segmentation on ADE20k, and class-conditional image generation on ImageNet. Our experiments demonstrate that Spiral RoPE consistently improves performance over baselines and prior 2D RoPE methods.

#### 4.1. Image Classification

**Setup.** We evaluate Spiral RoPE on ImageNet-1k (Deng et al., 2009) classification using DeiT (Touvron et al., 2021) as the base architecture. We train DeiT-Small, DeiT-Base and DeiT-Large models with and without Spiral RoPE on three target resolutions:  $224 \times 224$ ,  $384 \times 384$ , and  $448 \times 448$ . Following (Heo et al., 2024), we combine Spiral RoPE with the original absolute positional embedding (APE), as this combination has been shown to be effective. We use K = 16 directions for Spiral RoPE with all frequencies  $\theta_t$  scaled by a factor of 1.5 (i.e.,  $\theta_t = 1.5\theta_{\text{base}}^{-t/(d/4)}$ ). Models are pre-trained at  $224 \times 224$  resolution for 300 epochs (Small and Base) or 200 epochs (Large), then fine-tuned at the target resolution for 20 epochs. We compare against the standard DeiT baseline and RoPE-Mixed (Heo et al., 2024), which uses learnable per-head frequencies, in contrast to our method that uses pre-specified directions and frequencies. For evaluation, we report Top-1 accuracy on the validation set using the final checkpoint with exponential moving average (EMA) weights and full crop (crop ratio = 1.0). Additional training details are provided in Appendix A.

**Results.** Table 1 presents the classification results across different resolutions. Spiral RoPE consistently outperforms both the APE baseline and RoPE-Mixed across all configurations. These results demonstrate that the multi-directional encoding in Spiral RoPE provides richer positional information that benefits image classification, with larger improvements observed at higher resolutions.

**Extrapolation Robustness.** We further evaluate robustness to input resolutions unseen during training. Using models trained and fine-tuned at  $224 \times 224$  resolution, we evaluate performance across resolutions ranging from  $144 \times 144$  to  $512 \times 512$ . Other evaluation settings are the same as before. For resolution extrapolation, APE employs bicubic interpolation to resize the learned position vectors from training to evaluation resolution. RoPE-based methods ex-

Table 1. ImageNet-1k classification accuracy (%). Models are pretrained at  $224 \times 224$  and fine-tuned at each target resolution. Spiral RoPE consistently outperforms baselines across all configurations.

| Model  | Метнор             | 224          | 384          | 448          |
|--------|--------------------|--------------|--------------|--------------|
| DEIT-S | APE                | 79.11        | 81.88        | 82.24        |
|        | ROPE-MIXED         | <b>80.48</b> | 82.64        | 82.99        |
|        | SPIRAL ROPE (OURS) | 80.39        | <b>83.04</b> | <b>83.15</b> |
| DeiT-B | APE                | 82.36        | 84.16        | 84.29        |
|        | ROPE-MIXED         | 83.15        | 84.73        | 84.84        |
|        | SPIRAL ROPE (OURS) | <b>83.39</b> | <b>85.04</b> | <b>85.19</b> |
| DEIT-L | APE                | 83.24        | 84.92        | 85.13        |
|        | ROPE-MIXED         | 83.51        | 85.14        | 85.40        |
|        | SPIRAL ROPE (OURS) | <b>83.97</b> | <b>85.48</b> | <b>85.67</b> |

tend position indices to the new grid size while preserving the same frequency patterns, enabling smooth extrapolation through the inherent continuity of trigonometric functions.

Figure 4 shows the extrapolation robustness results for both DeiT-Base and DeiT-Large models. Both Spiral RoPE and RoPE-Mixed consistently outperform APE across all resolutions and model sizes, demonstrating the advantage of rotary position embeddings for resolution generalization. At lower resolutions near the training resolution, Spiral RoPE and RoPE-Mixed perform comparably. However, at higher resolutions (e.g.,  $384 \times 384$  to  $512 \times 512$ ), our Spiral RoPE exhibits a clear advantage over RoPE-Mixed across both model sizes, suggesting that our method with pre-specified directions and fixed sinusoidal frequencies achieves better extrapolation than RoPE-Mixed with learned frequencies when evaluating on unseen resolutions.

## 4.2. Semantic Segmentation

**Setup.** We evaluate Spiral RoPE on semantic segmentation using the ADE20k dataset (Zhou et al., 2017; 2018), which contains 20,000 training images and 2,000 validation images across 150 semantic categories. We use UperNet (Xiao et al., 2018) as the segmentation framework with ViT training recipe, and we initialize the backbone with the ImageNet pre-trained weights at  $224 \times 224$ . The segmentation models are trained at  $512 \times 512$  resolution for 160k iterations with a batch size of 16. For evaluation, we perform single-scale testing on the validation set and report mean Intersection over Union (mIoU), mean class accuracy (mAcc), and overall pixel accuracy (aAcc). Additional training details are provided in Appendix A.

**Results.** Table 2 summarizes semantic segmentation results on ADE20k, where Spiral RoPE consistently achieves significant improvements over the original APE baselines. Remarkably, we observe that the DeiT-Large model attains a notable mIoU of 49.12%, substantially outperforming

![](_page_5_Figure_8.jpeg)

Figure 4. Multi-resolution performance on ImageNet-1k. Models are trained at  $224 \times 224$  and evaluated across resolutions from  $144 \times 144$  to  $512 \times 512$ . Spiral RoPE demonstrates superior robustness to resolution changes compared to APE and RoPE-Mixed baselines across both model sizes.

*Table 2. ADE20k semantic segmentation results.* All models are pretrained on ImageNet-1k and then fintuned with an UperNet segmentation head for 160k iterations at  $512 \times 512$  resolution.

| BACKBONE | Метнор      | мІоИ         | мАсс         | AACC         |
|----------|-------------|--------------|--------------|--------------|
| DEIT-S   | APE         | 43.72        | 54.33        | 81.07        |
|          | Spiral RoPE | <b>45.44</b> | <b>56.33</b> | <b>81.61</b> |
| DEIT-B   | APE         | 46.89        | 57.28        | 82.30        |
|          | Spiral RoPE | <b>48.11</b> | <b>59.07</b> | <b>82.76</b> |
| DEIT-L   | APE         | 46.91        | 57.05        | 82.36        |
|          | Spiral RoPE | <b>49.12</b> | <b>60.11</b> | <b>83.40</b> |

the original baseline of 46.91% at virtually no additional computational cost. Moreover, the gains brought by Spiral RoPE on the ADE20K semantic segmentation task are intuitively more pronounced than those observed in classification (e.g., 3.06% vs. 0.73% accuracy improvement), suggesting that for dense prediction tasks such as semantic segmentation where reasoning is more complex and higher input resolutions are required, a well-designed relative position embedding can more effectively facilitate the modeling of spatial dependencies, demonstrating that Spiral RoPE consistently delivers larger benefits on more challenging tasks.

Table 3. Class-conditional image generation on ImageNet  $256 \times 256$  using DiT. We compare models trained for 400k steps with the standard training hyperparameters of DiT. We generate 50k samples using DDPM sampling with 250 steps and classifier-free guidance scale of 1.0. Baseline results with supermarks \* are taken from U-DiTs (Tian et al., 2024).

| MODEL    | МЕТНОО              | FID↓                  | sFID↓                 | IS↑                   | PREC.↑                | REC.↑                 |
|----------|---------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| DiT-S/2  | APE*<br>Spiral RoPE | 67.40<br><b>63.33</b> | 11.93<br><b>11.91</b> |                       | 0.368<br><b>0.386</b> | 0.559<br><b>0.574</b> |
| DiT-B/2  | APE*<br>Spiral RoPE | 42.84<br><b>37.74</b> | 8.24<br><b>8.23</b>   | 33.66<br><b>38.31</b> | 0.491<br><b>0.520</b> | 0.629<br><b>0.630</b> |
| DiT-L/2  | APE*<br>Spiral RoPE | 23.27<br><b>19.02</b> | <b>6.35</b> 6.36      | 59.63<br><b>69.49</b> | 0.611<br><b>0.638</b> | <b>0.635</b> 0.632    |
| DiT-XL/2 | APE*<br>Spiral RoPE | 20.05<br><b>15.55</b> | 6.25<br><b>5.87</b>   | 66.74<br><b>79.67</b> | 0.632<br><b>0.662</b> | 0.629<br><b>0.631</b> |

## 4.3. Image Generation

**Setup.** We also evaluate Spiral RoPE on class-conditional image generation using Diffusion Transformers (DiT) (Peebles & Xie, 2023). We train DiT models of various sizes (S, B, L, XL) with patch sizes of 2 (denoted as /2 in the original paper) on ImageNet  $256 \times 256$ . The /2 models process  $128 \times 128$  latent patches  $(16 \times 16$  tokens after patchification). We use K = 8 directions for S/B/L models and K = 6 for XL models. All models are trained for 400k steps with the original training recipe of DiT (Peebles & Xie, 2023). For evaluation, we generate 50k samples using DDPM sampling with 250 steps and classifier-free guidance scale of 1.0, and compute Fréchet Inception Distance (FID) (Heusel et al., 2018) against the ImageNet validation set statistics. Additional training details are provided in Appendix A.

**Results.** Table 3 presents the FID scores. The results suggest that Spiral RoPE is most effective when the sequence length is sufficiently large for the multi-directional encoding to capture meaningful spatial relationships.

**Extended Training.** To further demonstrate the effectiveness of Spiral RoPE when scaled to longer training, we conduct experiments following the setup of SiT (Ma et al., 2024). Specifically, we train the XL/2 models for 7 million steps on ImageNet  $256 \times 256$  and evaluate using classifier-free guidance with scale 1.5. Table 4 compares our Spiral RoPE against the original DiT baseline and SiT. Our Spiral RoPE achieves an FID of 1.74, outperforming both DiT (2.27) and SiT (2.06). Figure 5 shows qualitative samples generated by our model.

## 5. Analysis

In this section, we provide further analysis of Spiral RoPE, including visualization of attention maps and ablation studies on key hyperparameters.

Table 4. Extended training comparison on ImageNet  $256 \times 256$ . The Spiral RoPE model uses DiT-XL/2 architecture trained for 7M steps. We use classifier-free guidance with scale 1.5. Baseline results are from SiT (Ma et al., 2024).

| Метнор             | FID↓ | sFID↓ | IS↑    | PREC.↑ | REC.↑ |
|--------------------|------|-------|--------|--------|-------|
| DIT                | 2.27 | 4.60  | 278.24 | 0.83   | 0.57  |
| SIT                | 2.06 | 4.49  | 277.50 | 0.83   | 0.59  |
| SPIRAL ROPE (OURS) | 1.74 | 4.49  | 279.64 | 0.83   | 0.60  |

![](_page_6_Figure_11.jpeg)

Figure 5. Generated samples from our DiT-XL/2 model with Spiral RoPE trained for 7M steps on ImageNet  $256 \times 256$ . Images are generated using classifier-free guidance with scale 4.0.

## 5.1. Visualization of Attention Maps

To qualitatively understand the differences between positional encoding methods, we visualize attention maps from DeiT-Base models fine-tuned at  $448 \times 448$  resolution. We compare three variants: the original APE baseline, APE with axial 2D RoPE, and APE with Spiral RoPE (K=16).

**Class-Token Attention.** We first visualize the class-token attention by averaging the attention weights over all heads in the last transformer layer. Figure 6 shows representative examples from the ImageNet validation set. A clear progression is visible across the three methods. The APE baseline produces diffuse attention patterns, with substantial activation spread across both foreground objects and background regions. Axial RoPE improves over APE by allocating stronger attention to semantically relevant regions; however, its attention maps still exhibit noticeable residual activations in surrounding background areas and irrelevant objects. In contrast, Spiral RoPE yields the most object-centric and spatially coherent attention patterns, characterized by concentrated responses on the foreground objects and effective suppression of background activations. This advantage is particularly pronounced in multi-instance scenes, where Spiral RoPE sharply attends to multiple relevant objects, while Axial RoPE still exhibits axis-aligned artifacts and attention

![](_page_7_Figure_1.jpeg)

Figure 6. Class-token attention visualization. Each row shows the input image followed by attention maps from APE, Axial 2D RoPE, and Spiral RoPE. Attention is averaged over all heads in the last layer. Spiral RoPE produces more object-centric attention with cleaner background suppression compared to both counterparts.

leakage into nearby regions.

**Query-Token Attention.** To further examine how positional encoding influences local spatial relationships, we visualize attention patterns originating from individual patch tokens. Figure 3 shows representative examples in which the query token (marked by a red box) is located on a foreground object. Across all examples, APE and Axial RoPE produce attention maps that are broadly distributed across the image, with substantial activation on background regions and unrelated objects. In contrast, Spiral RoPE consistently produces markedly sharper and more spatially coherent attention patterns, with strong responses concentrated around the query location and semantically related regions. This suggests that Spiral RoPE facilitates more precise spatial alignment between query and key tokens, enabling attention to better respect local object boundaries and spatial proximity. We attribute this behavior to the multi-directional nature of Spiral RoPE, which encodes spatial relationships beyond axis-aligned directions and thus supports more flexible twodimensional spatial interactions.

#### 5.2. Ablation Studies

We conduct ablation studies on the ImageNet-1k classification task using DeiT-Base at  $224 \times 224$  resolution to analyze the effect of key hyperparameters in Spiral RoPE. All ablation experiments use the same training recipe as described in Section 4.1, with evaluation using the final checkpoint with EMA weights and full crop.

**Number of Directions** K. We first ablate the number of directions K in Spiral RoPE with the frequency scal-

Table 5. Ablation on the number of directions K.

*Table 6.* Ablation on the frequency scaling factor.

| K              | Acc (%) | SCA | ALE | Acc (%) |
|----------------|---------|-----|-----|---------|
| 2 (AXIAL ROPE) | 83.31   | 1.0 | 0   | 83.33   |
| 4              | 83.31   | 1.2 | 5   | 83.13   |
| 8              | 83.34   | 1.5 | 0   | 83.39   |
| 16             | 83.39   | 1.7 | 5   | 83.34   |
| 32             | 83.32   | 2.0 | 0   | 83.23   |

ing factor fixed at 1.5. As shown in Table 5, Spiral RoPE consistently outperforms the axial RoPE baseline (83.31%) across all values of K. Increasing K from 4 to 16 gradually improves accuracy, with K=16 achieving the best performance (83.39%). However, further increasing K to 32 slightly degrades performance (83.32%), suggesting that excessive directional partitioning may reduce the embedding capacity per direction without providing additional benefit. Based on these results, we select K=16 as the default configuration for our classification experiments.

Frequency Scaling Factor. We also ablate the frequency scaling factor that multiplies all base frequencies  $\theta_t$  in Spiral RoPE, with K fixed at 16. As shown in Table 6, the scaling factor has a notable impact on performance. A scaling factor of 1.0 (no scaling) achieves 83.33%, while increasing to 1.5 yields the best result (83.39%). However, larger scaling factors (1.75, 2.0) lead to decreased performance, likely because excessively high frequencies cause the rotations to vary too rapidly across positions, making it difficult for the model to learn stable positional relationships. We select 1.5 as the default scaling factor for our experiments.

## 6. Conclusion

We have presented Spiral RoPE, a simple yet effective extension of rotary position embedding for vision transformers that addresses the directional constraint of conventional Axial 2D RoPE by distributing positional encoding across K uniformly spaced directions. Extensive experiments across image classification, semantic segmentation, and image generation demonstrate consistent performance improvements, with particularly strong resolution extrapolation capability. These results suggest that directional diversity in positional encoding plays a critical role in modeling 2D spatial relationships, and that modest geometric extensions to rotary embeddings can substantially enhance spatial representation in vision transformers.

#### References

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., Hui, B., Ji, L., Li, M., Lin, J., Lin, R., Liu, D., Liu, G., Lu, C., Lu, K., et al.

- Qwen Technical Report, September 2023. URL http://arxiv.org/abs/2309.16609.
- Chen, S., Wong, S., Chen, L., and Tian, Y. Extending Context Window of Large Language Models via Positional Interpolation, June 2023. URL http://arxiv.org/abs/2306.15595.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, June 2009. doi: 10.1109/CVPR.2009.5206848. URL https://ieeexplore.ieee.org/document/5206848.
- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, May 2019. URL http://arxiv.org/abs/1810.04805.
- Dhariwal, P. and Nichol, A. Diffusion Models Beat GANs on Image Synthesis, June 2021. URL http://arxiv.org/abs/2105.05233.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, June 2021. URL http://arxiv.org/abs/2010.11929.
- Fang, Y., Sun, Q., Wang, X., Huang, T., Wang, X., and Cao, Y. EVA-02: A Visual Representation for Neon Genesis. *Image and Vision Computing*, 149:105171, September 2024. ISSN 02628856. doi: 10.1016/j.imavis.2024.105171. URL http://arxiv.org/abs/2303.11331.
- Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., et al. The Llama 3 Herd of Models, November 2024. URL http://arxiv.org/abs/2407.21783.
- Heo, B., Park, S., Han, D., and Yun, S. Rotary Position Embedding for Vision Transformer. In *Computer Vision ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part X*, pp. 289–305, Berlin, Heidelberg, November 2024. Springer-Verlag. ISBN 978-3-031-72683-5. doi: 10.1007/978-3-031-72684-2\_17. URL https://doi.org/10.1007/978-3-031-72684-2\_17.

- Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, January 2018. URL http://arxiv.org/abs/1706.08500.
- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, August 2021. URL http://arxiv.org/abs/2103.14030.
- Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., and Kembhavi, A. Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action, December 2023. URL http://arxiv.org/abs/2312.17172.
- Lu, Z., Wang, Z., Huang, D., Wu, C., Liu, X., Ouyang, W., and Bai, L. FiT: Flexible Vision Transformer for Diffusion Model, October 2024. URL http://arxiv.org/abs/2402.12376.
- Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M., Vanden-Eijnden, E., and Xie, S. SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers, September 2024. URL http://arxiv.org/abs/2401.08740.
- Peebles, W. and Xie, S. Scalable Diffusion Models with Transformers, March 2023. URL http://arxiv.org/abs/2212.09748.
- Peng, B., Quesnelle, J., Fan, H., and Shippole, E. YaRN: Efficient Context Window Extension of Large Language Models, November 2023. URL http://arxiv.org/abs/2309.00071.
- Press, O., Smith, N. A., and Lewis, M. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, April 2022. URL http: //arxiv.org/abs/2108.12409.
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, September 2023. URL http://arxiv.org/abs/1910.10683.
- Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., and Feichtenhofer, C. SAM 2: Segment Anything in Images and Videos, October 2024. URL http://arxiv.org/abs/2408.00714.
- Shaw, P., Uszkoreit, J., and Vaswani, A. Self-Attention with Relative Position Representations, April 2018. URL http://arxiv.org/abs/1803.02155.

- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. RoFormer: Enhanced Transformer with Rotary Position Embedding, November 2023. URL http://arxiv.org/abs/2104.09864.
- Tian, Y., Tu, Z., Chen, H., Hu, J., Xu, C., and Wang, Y. U-DiTs: Downsample Tokens in U-Shaped Diffusion Transformers, October 2024. URL http://arxiv.org/abs/2405.02730.
- Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and Jégou, H. Training data-efficient image transformers & distillation through attention, January 2021. URL http://arxiv.org/abs/2012.12877.
- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. LLaMA: Open and Efficient Foundation Language Models, February 2023a. URL http://arxiv.org/abs/2302.13971.
- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., et al. Llama 2: Open Foundation and Fine-Tuned Chat Models, July 2023b. URL http://arxiv.org/abs/2307.09288.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention Is All You Need, August 2023. URL http://arxiv.org/abs/1706.03762.
- Wang, C., Guo, J., Li, H., Tian, Y., Nie, Y., Xu, C., and Han, K. Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models, October 2025. URL http://arxiv.org/abs/2505. 16416.
- Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., and Lin, J. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution, September 2024. URL http://arxiv.org/abs/2409.12191.
- Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., et al. Qwen-Image Technical Report, August 2025. URL http://arxiv.org/abs/2508.02324.
- Xiao, T., Liu, Y., Zhou, B., Jiang, Y., and Sun, J. Unified Perceptual Parsing for Scene Understanding, July 2018. URL http://arxiv.org/abs/1807.10221.

- Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J., et al. Qwen2 Technical Report, September 2024. URL http://arxiv.org/abs/2407.10671.
- Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., et al. Qwen3 Technical Report, May 2025a. URL http://arxiv.org/abs/2505.09388.
- Yang, A., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., et al. Qwen2.5 Technical Report, January 2025b. URL http://arxiv.org/abs/2412.15115.
- Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., and Torralba, A. Scene Parsing through ADE20K Dataset. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5122–5130, July 2017. doi: 10. 1109/CVPR.2017.544. URL https://ieeexplore.ieee.org/document/8100027.
- Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., and Torralba, A. Semantic Understanding of Scenes through the ADE20K Dataset, October 2018. URL http://arxiv.org/abs/1608.05442.

# A. Implementation Details

## A.1. Image Classification

**Pre-training.** We train DeiT models on ImageNet-1K following an improved training recipe. For DeiT-Base and DeiT-Small, we train for 300 epochs with a batch size of 4096, learning rate of  $2 \times 10^{-4}$ , weight decay of 0.3, and drop path rate of 0.1. For DeiT-Large, we train for 200 epochs with the same hyperparameters except drop path rate of 0.2. We use AdamW optimizer with  $\beta_1 = 0.95$  and cosine learning rate schedule with 20 warmup epochs. We employ the same data augmentation pipeline during training as the original DeiT implementation.

**Fine-tuning.** For fine-tuning at  $224 \times 224$ , we train for 20 epochs with learning rate  $10^{-5}$ , weight decay 0.1, and the same drop path rate as pre-training. For higher resolutions ( $384 \times 384$  and  $448 \times 448$ ), we fine-tune from the  $224 \times 224$  checkpoint for 20 epochs with the same hyperparameters. We use full crop (crop ratio 1.0) during evaluation.

**Spiral RoPE Configuration.** For all classification experiments, we use K=16 directions with base frequency scaling factor 1.5 (i.e.,  $\theta_{\text{base}}=15000$ ). Spiral RoPE is combined with the original absolute positional embedding (APE).

| Hyperparameter             | DeiT-Base and DeiT-Small | DeiT-Large         |  |  |  |
|----------------------------|--------------------------|--------------------|--|--|--|
| Epochs                     | 300                      | 200                |  |  |  |
| Batch size                 | 4096                     | 4096               |  |  |  |
| Optimizer                  | AdamW                    | AdamW              |  |  |  |
| Learning rate              | $2 \times 10^{-4}$       | $2 \times 10^{-4}$ |  |  |  |
| Weight decay               | 0.3                      | 0.3                |  |  |  |
| $\beta_1, \beta_2$         | 0.95, 0.999              | 0.95, 0.999        |  |  |  |
| LR schedule                | Cosine                   | Cosine             |  |  |  |
| Warmup epochs              | 20                       | 20                 |  |  |  |
| Drop path                  | 0.1                      | 0.2                |  |  |  |
| Label smoothing            | 0.1                      | 0.1                |  |  |  |
| Data Augmentation          |                          |                    |  |  |  |
| RandAugment                | (9, 0.5)                 | (9, 0.5)           |  |  |  |
| Mixup $\alpha$             | 0.8                      | 0.8                |  |  |  |
| CutMix $\alpha$            | 1.0                      | 1.0                |  |  |  |
| Random erasing             | 0.25                     | 0.25               |  |  |  |
| Spiral RoPE Parameters     |                          |                    |  |  |  |
| Number of directions $(K)$ | 16                       | 16                 |  |  |  |
| Base frequency scale       | 1.5                      | 1.5                |  |  |  |

Table 7. Pre-training hyperparameters for ImageNet-1K classification.

## A.2. Semantic Segmentation

We use UperNet as the segmentation framework with DeiT backbones initialized from ImageNet pre-trained weights at  $224 \times 224$  resolution. Models are trained on ADE20K at  $512 \times 512$  resolution for 160K iterations with a total batch size of 16 (8 GPUs  $\times$  2 samples per GPU). We use AdamW optimizer with learning rate  $6 \times 10^{-5}$ , weight decay 0.01, and polynomial learning rate decay with 1.5K warmup iterations. Weight decay is disabled for position embeddings, class tokens, normalization layers, and layer scale parameters. For Spiral RoPE models, we use the same K=16 and base frequency scale 1.5 as in classification.

#### A.3. Image Generation

We follow the training recipe of the original DiT implementation (Peebles & Xie, 2023). Specifically, we train DiT models on ImageNet  $256 \times 256$  for class-conditional image generation. All models are trained for 400K steps with a global batch

size of 256. We use the AdamW optimizer with a constant learning rate of  $10^{-4}$  and no weight decay. We employ the EMA of model weights with decay 0.9999 for evaluation. For sampling, we use DDPM with 250 steps and classifier-free guidance scale of 1.0. FID is computed on 50K generated samples using the ADM evaluation toolkit (Dhariwal & Nichol, 2021).

**Spiral RoPE Configuration.** For DiT experiments, we use K=8 for S, B, and L models, and K=6 for XL models. The base frequency scale is set to 1.5 for all models.

Table 8. DiT training hyperparameters.

| Hyperparameter               | Value              |  |  |  |  |
|------------------------------|--------------------|--|--|--|--|
| Training steps               | 400K               |  |  |  |  |
| Global batch size            | 256                |  |  |  |  |
| Optimizer                    | AdamW              |  |  |  |  |
| Learning rate                | $1 \times 10^{-4}$ |  |  |  |  |
| Weight decay                 | 0.0                |  |  |  |  |
| EMA decay                    | 0.9999             |  |  |  |  |
| Sampling                     | Sampling           |  |  |  |  |
| Sampling steps               | 250                |  |  |  |  |
| CFG scale                    | 1.0                |  |  |  |  |
| Spiral RoPE Parameters       |                    |  |  |  |  |
| K (S/B/L)                    | 8                  |  |  |  |  |
| $K\left( \mathrm{XL}\right)$ | 6                  |  |  |  |  |
| Base frequency scale         | 1.5                |  |  |  |  |
|                              |                    |  |  |  |  |

# **B.** Additional Generated Samples

We provide additional uncurated samples generated by our DiT-XL/2 model with Spiral RoPE, trained for 7 million steps on ImageNet  $256 \times 256$ . For each class, we show 40 randomly generated samples. All samples are generated using classifier-free guidance with scale 4.0.

![](_page_12_Picture_1.jpeg)

Figure 7. Class 207: Golden Retriever

![](_page_12_Picture_3.jpeg)

Figure 8. Class 281: Tabby Cat

![](_page_13_Picture_1.jpeg)

Figure 9. Class 279: Arctic Fox

![](_page_13_Picture_3.jpeg)

Figure 10. Class 360: Otter

![](_page_14_Picture_1.jpeg)

Figure 11. Class 33: Loggerhead Turtle

![](_page_14_Picture_3.jpeg)

Figure 12. Class 88: Macaw

![](_page_15_Picture_1.jpeg)

Figure 13. Class 323: Monarch Butterfly

![](_page_15_Picture_3.jpeg)

Figure 14. Class 417: Balloon

![](_page_16_Picture_1.jpeg)

Figure 15. Class 979: Valley

![](_page_16_Picture_3.jpeg)

Figure 16. Class 980: Volcano