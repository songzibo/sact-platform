# HuggingFace Transformers 模型激活函数综合分析报告

## 📊 Top 3 激活函数使用统计

| 排名 | 激活函数 | 模型数量 | 占比 |
|:---:|:-------:|:-------:|:----:|
| 1 | **GELU** | 173 | 48.6% |
| 2 | **SiLU** | 98 | 27.5% |
| 3 | **ReLU** | 25 | 7.0% |

---

## 📋 激活函数完整分类表

| 激活函数类型 | 激活函数名 | 模型数 | 代表模型(前5个) |
|:----------:|:---------|:-----:|:--------------|
| GELU类 | gelu | 173 | align, altclip, audioflamingo3, autoformer, beit |
| SiLU类 | silu | 98 | afmoe, aimv2, aria, bamba, blt |
| ReLU类 | relu | 25 | bit, cohere_asr, conditional_detr, decision_transformer |
| GELU变体 | gelu_pytorch_tanh | 14 | gemma, gpt_bigcode, idefics2, idefics3, minicpmv4_6 |
| GELU变体 | quick_gelu | 12 | clip, clipseg, ernie4_5_vl_moe, git, groupvit |
| GELU变体 | gelu_new | 11 | albert, big_bird, bigbird_pegasus, codegen, fnet |
| ReLU变体 | relu2 | 8 | arcee, bitnet, fuyu, jais2, nanochat |
| 其他 | swish | 5 | efficientnet, mobilevitv2, seamless_m4t, seamless_m4t_v2 |
| 其他 | hardswish | 3 | pp_lcnet, pp_lcnet_v3, slanet |

---

## 📑 各模型激活函数详细列表

### GELU (173个模型)
align, altclip, audio_spectrogram_transformer, audioflamingo3, autoformer, beit, bert, bert_generation, biogpt, blenderbot, blenderbot_small, blip, blip_2, bridgetower, bros, camembert, canine, chinese_clip, clap, clvp, convbert, convnext, convnextv2, d_fine, deberta, deberta_v2, deimv2, deit, dinat, dinov2, dinov2_with_registers, dinov3_convnext, dinov3_vit, dpr, dpt, edgetam, edgetam_video, electra, eomt, eomt_dinov3, ernie, fast_vlm, flava, florence2, focalnet, glm_image, glmasr, glpn, got_ocr2, gpt_neox, gpt_neox_japanese, hiera, hubert, ibert, idefics, ijepa, informer, instructblip, instructblipvideo, internvl, janus, jina_embeddings_v3, kosmos2_5, layoutlm, layoutlmv2, layoutlmv3, layoutxlm, led, lfm2_vl, lightglue, lilt, llama4, llava, llava_next, llava_next_video, llava_onevision, longformer, luke, lw_detr, lxmert, marian, markuplm, mbart, megatron_bert, mimi, mistral3, mlcd, mllama, moonshine, moonshine_streaming, mpnet, mra, musicflamingo, musicgen, musicgen_melody, mvp, patchtst, pegasus, pegasus_x, perceiver, pixio, pixtral, plbart, poolformer, pp_chart2table, pp_doclayout_v2, pp_doclayout_v3, pp_formulanet, prophetnet, pvt, pvt_v2, qianfan_ocr, qwen2_audio, rembert, rf_detr, roberta, roberta_prelayernorm, roc_bert, roformer, rt_detr, rt_detr_v2, sam, sam2, sam2_video, sam3, sam3_tracker, sam3_tracker_video, sam_hq, segformer, seggpt, slanext, speecht5, splinter, squeezebert, swiftformer, swin, swin2sr, swinv2, tapas, time_series_transformer, timesformer, trocr, tvp, unispeech, unispeech_sat, vibevoice_acoustic_tokenizer, video_llava, videomae, videomt, vilt, vipllava, visual_bert, vit, vit_mae, vit_msn, vitdet, vitpose_backbone, vjepa2, voxtral, wav2vec2, wav2vec2_conformer, wavlm, whisper, xglm, xlm_roberta, xlm_roberta_xl, xmod, yolos, yoso, zamba, zamba2, zoedepth

### SiLU (98个模型)
afmoe, aimv2, aria, bamba, blt, chameleon, cohere, cohere2, csm, cwm, deepseek_v2, deepseek_v3, deepseek_v4, dia, diffllama, doge, dots1, emu3, ernie4_5, ernie4_5_moe, eurobert, evolla, exaone4, exaone4_5, exaone_moe, falcon_h1, falcon_mamba, flex_olmo, gemma4, glm, glm4, glm4_moe, glm4_moe_lite, glm4v, glm4v_moe, glm_moe_dsa, glm_ocr, gpt_oss, granite, granite4_vision, granitemoe, granitemoehybrid, granitemoeshared, helium, higgs_audio_v2, hunyuan_v1_dense, hunyuan_v1_moe, hy_v3, hyperclovax, jamba, jetmoe, kyutai_speech_to_text, laguna, lasr, llama, longcat_flash, mamba, mamba2, minimax, minimax_m2, ministral, ministral3, mistral, mistral4, mixtral, mobilevit, moshi, nomic_bert, olmo, olmo2, olmo3, olmo_hybrid, olmoe, ovis2, parakeet, pe_audio, pe_audio_video, pe_video, phi3, phimoe, pp_ocrv5_mobile_rec, pp_ocrv5_server_rec, qwen2, qwen2_5_omni, qwen2_5_vl, qwen2_moe, qwen3, qwen3_5, qwen3_5_moe, qwen3_moe, qwen3_next, qwen3_vl_moe, seed_oss, smollm3, solar_open, stablelm, voxtral_realtime, youtu

### ReLU (25个模型)
bit, cohere_asr, conditional_detr, decision_transformer, deformable_detr, detr, efficientloftr, fsmt, grounding_dino, hgnet_v2, m2m_100, mask2former, maskformer, mm_grounding_dino, mobilebert, nllb_moe, opt, pp_ocrv5_server_det, reformer, regnet, resnet, sam3_lite_text, speech_to_text, table_transformer, vits

### GELU变体

**gelu_pytorch_tanh (14):** gemma, gpt_bigcode, idefics2, idefics3, minicpmv4_6, paddleocr_vl, phi4_multimodal, qwen3_omni_moe, qwen3_vl, siglip, siglip2, smolvlm, starcoder2, video_llama_3

**quick_gelu (12):** clip, clipseg, ernie4_5_vl_moe, git, groupvit, imagegpt, kosmos2, metaclip_2, owlv2, owlvit, qwen2_vl, x_clip

**gelu_new (11):** albert, big_bird, bigbird_pegasus, codegen, fnet, funnel, gpt2, gpt_neo, gptj, nystromformer, phi

### ReLU变体

**relu2 (8):** arcee, bitnet, fuyu, jais2, nanochat, nemotron, nemotron_h, persimmon

**relu6 (2):** mobilenet_v1, mobilenet_v2

### 其他激活函数

**swish (5):** efficientnet, mobilevitv2, seamless_m4t, seamless_m4t_v2, wav2vec2_bert

**hardswish (3):** pp_lcnet, pp_lcnet_v3, slanet

**prelu (2):** dab_detr, uvdoc

**xielu (1):** apertus

---

## 💡 总结

1. **GELU仍是主导（48.6%）**：BERT、ViT、CLIP等经典Transformer模型都使用GELU
2. **现代大模型偏好SiLU（27.5%）**：LLaMA、Qwen、Mistral、DeepSeek等最新大语言模型都采用SiLU
3. **ReLU仍用于特定场景（7%）**：DETR、Decision Transformer等早期模型使用ReLU
4. **变体众多**：为适配不同硬件和效率需求，存在多种GELU和ReLU变体
5. **总计分析了356个模型，识别到14种激活函数**

---

---

# 附录：激活函数的数学公式与测试用例

## 1. GELU (Gaussian Error Linear Unit)

**数学公式：**
$$GELU(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}[1 + erf(\frac{x}{\sqrt{2}})]$$

其中 $\Phi(x)$ 是标准正态分布的CDF，$erf$ 是误差函数。

**近似计算（用于实际实现）：**
$$GELU(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$$

**使用此激活函数的模型（173个）：**
BERT, ViT, CLIP (使用quick_gelu变体), BLIP, CamemBERT, RoBERTa, DeBERTa,ALBERT, ELECTRA, FNet, Funnel, GPT-NeoX, Longformer, LXMert, mBART, MPNet, ProphetNet, Sam, Swin, TimeSformer, Whisper, XLM-RoBERTa等

**PyTorch 测试用例：**
```python
import torch
import torch.nn.functional as F
import math

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = F.gelu(x)
print("GELU input:", x)
print("GELU output:", output)
# 输出: tensor([-0.0455, -0.1588, 0.0000, 0.8412, 1.9545])
```

**NumPy 测试用例：**
```python
import numpy as np
from scipy.stats import norm

def gelu(x):
    return x * norm.cdf(x)

def gelu_fast(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("GELU (精确):", gelu(x))
print("GELU (近似):", gelu_fast(x))
```

---

## 2. SiLU (Sigmoid Linear Unit) / Swish

**数学公式：**
$$SiLU(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} = x \cdot sigmoid(x)$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是Sigmoid函数。

**性质：**
- 输出范围：(-0.278, ∞)
- 非单调：当 x < 0 时有最小值约 -0.278
- 自门控特性：输入决定门控值

**使用此激活函数的模型（98个）：**
LLaMA, LLaMA2, LLaMA3, LLaMA4, Mistral, Mixtral, Qwen, Qwen2, Qwen3, DeepSeek, Gemma, Phi-3, Falcon, Mamba, Jamba, Granite, OLMo,StableLM, Chameleon, Cohere, GLM4, Yi, Exaone等

**PyTorch 测试用例：**
```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = F.silu(x)
print("SiLU input:", x)
print("SiLU output:", output)
# 输出: tensor([-0.2387, -0.2689, 0.0000, 0.7311, 1.9547])
```

**NumPy 测试用例：**
```python
import numpy as np

def silu(x):
    return x / (1 + np.exp(-x))

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("SiLU output:", silu(x))
# 输出: [-0.2387 -0.2689  0.      0.7311  1.9547]
```

---

## 3. ReLU (Rectified Linear Unit)

**数学公式：**


$$
ReLU(x) = \max(0, x) = \begin{cases} 
x & \text{if } x > 0 \\ 
0 & \text{if } x \leq 0 
\end{cases}
$$

**性质：**
- 输出范围：[0, ∞)
- 计算简单，梯度恒定为0或1
- 可能产生"死亡神经元"问题

**使用此激活函数的模型（25个）：**
DETR, Deformable DETR, Conditional DETR, Decision Transformer, MobileBERT, OPT, FSMT, Reformer, Speech2Text, VITS, MaskFormer, Mask2Former, Grounding DINO等

**PyTorch 测试用例：**
```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = F.relu(x)
print("ReLU input:", x)
print("ReLU output:", output)
# 输出: tensor([0., 0., 0., 1., 2.])
```

**NumPy 测试用例：**
```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("ReLU output:", relu(x))
# 输出: [0. 0. 0. 1. 2.]
```

---

# 附录2：激活函数变体详解

## 1. GELU变体

### 1.1 GELU-PyTorch-Tanh（14个模型）

**数学公式：**
$$GELU(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$$



**使用此激活函数的模型：**
Gemma, GPT-BigCode, Idefics2, Idefics3, MiniCPMv4_6, PaddleOCR-VL, Phi4-Multimodal, Qwen3-Omni-MoE, Qwen3-VL, SigLIP, SigLIP2, SmolVLM, StarCoder2, Video-Llama3

**代码实现：**
```python
def gelu_pytorch_tanh(x):
    return x * 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
```

### 1.2 Quick-GELU（12个模型）

**数学公式：**
$$QuickGELU(x) = x \cdot \sigma(1.702 \cdot x)$$

其中 $\sigma$ 是Sigmoid函数，比标准GELU计算更快。

**使用此激活函数的模型：**
CLIP, CLIPSeg, Ernie4.5-VL-MoE, GIT, GroupViT, ImageGPT, Kosmos2, MetaCLIP2, OwlV2, OwlViT, Qwen2-VL, X-CLIP

**代码实现：**
```python
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)
```

### 1.3 GELU-New（11个模型）

**数学公式：**
$$GELU_{new}(x) = 0.5x(1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$$

原始BERT论文中使用的近似实现。

**使用此激活函数的模型：**
ALBERT, BigBird, BigBird-Pegasus, CodeGen, FNet, Funnel, GPT-2, GPT-Neo, GPT-J, Nystromformer, Phi

**代码实现：**
```python
def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
```

### 1.4 GELU-Fast / GELU-Python（2个模型）

**数学公式：**
$$GELU(x) = 0.5x(1 + erf(\frac{x}{\sqrt{2}}))$$

使用erf的精确实现。

**使用此激活函数的模型：**
Sew, ViViT

**代码实现：**
```python
def gelu_fast(x):
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))
```

---

## 2. ReLU变体

### 2.1 ReLU² (Squared ReLU)（8个模型）

**数学公式：**
$$ReLU^2(x) = (max(0, x))^2 = \begin{cases} x^2 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

用于BitNet等量化模型，可以更好地保持量化精度。

**使用此激活函数的模型：**
Arcee, BitNet, Fuyu, Jais2, NanoChat, Nemotron, Nemotron-H, Persimmon

**代码实现：**
```python
def relu2(x):
    return torch.relu(x) ** 2
```

### 2.2 ReLU6（2个模型）

**数学公式：**
$$ReLU6(x) = min(max(0, x), 6)$$

限制输出到 [0, 6] 范围，用于移动端优化。

**使用此激活函数的模型：**
MobileNet-V1, MobileNet-V2

**代码实现：**
```python
def relu6(x):
    return torch.clamp(torch.relu(x), max=6)
```

---

## 3. 其他激活函数

### 3.1 Swish（5个模型）

**数学公式：**
$$Swish_\beta(x) = x \cdot \sigma(\beta \cdot x)$$

其中 $\beta$ 可学习（通常设为1）。

**使用此激活函数的模型：**
EfficientNet, MobileViTV2, Seamless-M4T, Seamless-M4T-V2, Wav2Vec2-BERT

**代码实现：**
```python
def swish(x, beta=1):
    return x * torch.sigmoid(beta * x)
```

### 3.2 HardSwish（3个模型）

**数学公式：**
$$HardSwish(x) = x \cdot \frac{min(max(x+3, 0), 6}{6}$$

ReLU6风格的硬版本，MobileNet性能优化。

**使用此激活函数的模型：**
PP-LCNet, PP-LCNet-V3, SLANet

**代码实现：**
```python
def hardswish(x):
    return x * torch.clamp(x + 3, min=0, max=6) / 6
```

### 3.3 PReLU (Parametric ReLU)（2个模型）

**数学公式：**
$$PReLU(x) = \begin{cases} x & \text{if } x > 0 \\ a \cdot x & \text{if } x \leq 0 \end{cases}$$

其中 $a$ 是可学习的参数。

**使用此激活函数的模型：**
DAB-DETR, UVDoc

**代码实现：**
```python
class PReLU(torch.nn.PReLU):
    def __init__(self, num_parameters=1):
        super().__init__(num_parameters=num_parameters)
```

### 3.4 xielu（1个模型）

**数学公式：**



$$ \text{xIELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha \cdot (e^x - 1) & \text{if } x \leq 0 \end{cases} $$


ELU的指数变体，Alpha默认值为1.0。

**使用此激活函数的模型：**
Apertus

**代码实现：**
```python
def xielu(x, alpha=1.0):
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
```

---

## 变体函数对比表

| 激活函数 | 公式 | 计算复杂度 | 输出范围 | 模型数量 | 典型模型 |
|:------:|:-----|:-------:|:------:|:-------:|:------:|
| GELU | x·Φ(x) | 中等 | (-0.17, ∞) | 173 | BERT, ViT |
| GELU-PyTorch-Tanh | x·tanh(...) | 中等 | ~ | 14 | Gemma |
| Quick-GELU | x·σ(1.702x) | 低 | ~ | 12 | CLIP |
| GELU-New | x·tanh(...) | 中等 | ~ | 11 | GPT-2 |
| SiLU | x·σ(x) | 中等 | (-0.278, ∞) | 98 | LLaMA |
| ReLU | max(0,x) | 低 | [0, ∞) | 25 | DETR |
| ReLU² | max(0,x)² | 低 | [0, ∞) | 8 | BitNet |
| ReLU6 | min(max(0,x),6) | 低 | [0, 6] | 2 | MobileNet |
| Swish | x·σ(x) | 中等 | ~ | 5 | EfficientNet |
| HardSwish | x·clamp/6 | 低 | [-4.5, 6] | 3 | PP-LCNet |
| PReLU | max(ax, x) | 低 | R | 2 | DAB-DETR |
| xielu | exp(x)-1 | 中等 | R | 1 | Apertus |

---

## 变体函数对比表

| 激活函数 | 公式 | 计算复杂度 | 输出范围 | 使用场景 |
|:------:|:-----|:-------:|:------:|:------:|
| GELU | x·Φ(x) | 中等 | (-0.17, ∞) | BERT, ViT |
| SiLU | x·σ(x) | 中等 | (-0.278, ∞) | LLaMA, Qwen |
| ReLU | max(0,x) | 低 | [0, ∞) | 早期模型 |
| ReLU² | max(0,x)² | 低 | [0, ∞) | BitNet |
| ReLU6 | min(max(0,x),6) | 低 | [0, 6] | MobileNet |
| Quick-GELU | x·σ(kx) | 低 | ~ | CLIP |
| HardSwish | x·clamp(x+3)/6 | 低 | [-4.5, 6] | 轻量模型 |

---

*分析日期: 2026年5月15日*  
*数据来源: HuggingFace Transformers src/transformers/models/*
