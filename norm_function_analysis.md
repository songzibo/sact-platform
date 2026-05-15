# Transformers 中 RMS 归一化函数分析报告

> 说明：你这次要求的是 **RMS 归一化（RMSNorm）**，不是激活函数。以下内容基于当前仓库 `src/transformers/models` 的实现模式整理。

## 1. RMSNorm 的核心定义

给定输入向量 x（最后一维大小为 d），RMS 定义为：

`rms(x) = sqrt(mean(x^2) + eps)`

标准 RMSNorm 输出：

`y = x / rms(x) * w`

其中：
- `mean(x^2)` 按最后一维计算
- `eps` 是数值稳定项
- `w` 是可学习缩放参数（与最后一维同长度）

---

## 2. 仓库里主要的 RMS 归一化函数类型

### A. 主流标准型（Llama/T5 风格，最常见）

表达式：

`variance = mean(x^2, axis=-1, keepdims=True)`

`x_norm = x / sqrt(variance + eps)`

`y = w * x_norm`

实现特征：
- 常见写法是先转 float32 计算，再 cast 回原 dtype
- 类名通常是 `XXXRMSNorm`
- 大多数 LLM 都是这一类

仓库内统计（按模型目录粗统计）：
- 约 114 个模型目录属于该主流实现

典型模型：
- llama / mistral / mixtral / qwen2 / qwen3 / deepseek_v2 / deepseek_v3 / deepseek_v4（其标准分支）/ phi3 / granite / zamba 等

结论：
- 这是当前最主流、最标准的 RMSNorm 形态。

---

### B. Gemma 系列 `1 + weight` 缩放变体

表达式：

`variance = mean(x^2, axis=-1, keepdims=True)`

`x_norm = x / sqrt(variance + eps)`

`y = x_norm * (1 + w)`

与标准型差异：
- 标准型是乘 `w`
- 此变体乘 `1 + w`

仓库内模型（11 个目录）：
- gemma, gemma2, gemma3, recurrent_gemma, t5gemma, t5gemma2, vaultgemma, nemotron, qwen3_5, qwen3_5_moe, qwen3_next

结论：
- 属于有代表性的主流变体（在 Gemma 及部分新模型中常见），但总体覆盖面低于 Llama/T5 标准型。

---

### C. Gated RMSNorm（门控 RMS 归一化）

常见形式（简化）：

`g = gate(x)` 或外部给定门控张量

`x_g = x * g`

`variance = mean(x_g^2, axis=-1, keepdims=True)`

`y = x_g / sqrt(variance + eps) * w`

实际实现会按模块不同有细节差异（分组、核函数融合、是否先 norm 再 gate 等）。

仓库内模型（9 个目录）：
- bamba, falcon_h1, granitemoehybrid, mamba2, olmo_hybrid, qwen3_5, qwen3_5_moe, qwen3_next, zamba2

结论：
- 这是偏结构化/架构定制的变体，多见于状态空间模型或融合核优化路径，不是最普适默认形态。

---

### D. Unweighted RMSNorm（无可学习 weight）

表达式：

`y = x / sqrt(mean(x^2, axis=-1, keepdims=True) + eps)`

仓库内：
- 在 `deepseek_v4` 中可见 `DeepseekV4UnweightedRMSNorm`

结论：
- 特殊用途变体，不是主流。

---

### E. 其他特殊实现（精度/核融合导向）

例如：
- 某些实现中将 `weight` 与归一化结果相乘顺序微调（先乘后 cast）
- 某些模型使用 fused kernel 替代 Python 路径

这类更多属于工程优化，不改变 RMSNorm 的核心数学本质。

---

## 3. 哪些是主流，哪些是变种

主流：
1. **标准 Llama/T5 风格 RMSNorm**（最普遍）
2. **Gemma 的 `1 + w` 变体**（重要但次主流）

变种 / 特殊：
1. Gated RMSNorm
2. Unweighted RMSNorm
3. 各类 fused/核函数路径（实现优化型差异）

---

## 4. NumPy 实现（对应不同 RMSNorm 形态）

```python
import numpy as np


def _to_compute_dtype(x: np.ndarray) -> np.ndarray:
    # 常见实现会在 float32 做方差计算
    return x.astype(np.float32)


def rmsnorm_standard(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x_fp32 = _to_compute_dtype(x)
    var = np.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
    x_norm = x_fp32 / np.sqrt(var + eps)
    y = x_norm * weight.astype(np.float32)
    return y.astype(x.dtype, copy=False)


def rmsnorm_gemma_plus_one(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x_fp32 = _to_compute_dtype(x)
    var = np.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
    x_norm = x_fp32 / np.sqrt(var + eps)
    y = x_norm * (1.0 + weight.astype(np.float32))
    return y.astype(x.dtype, copy=False)


def rmsnorm_unweighted(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x_fp32 = _to_compute_dtype(x)
    var = np.mean(x_fp32 * x_fp32, axis=-1, keepdims=True)
    y = x_fp32 / np.sqrt(var + eps)
    return y.astype(x.dtype, copy=False)


def rmsnorm_gated(x: np.ndarray, gate: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x_fp32 = _to_compute_dtype(x)
    g_fp32 = gate.astype(np.float32)
    xg = x_fp32 * g_fp32
    var = np.mean(xg * xg, axis=-1, keepdims=True)
    xg_norm = xg / np.sqrt(var + eps)
    y = xg_norm * weight.astype(np.float32)
    return y.astype(x.dtype, copy=False)
```

---

## 5. 每个函数单独测试用例（覆盖 float32 / float16 / int32 + 多维度）

完整可运行测试代码见：`../activation_functions_test.py`

测试设计要点：
- 每个函数独立 `test_xxx()`
- 覆盖 `float32`, `float16`, `int32` 输入
- 覆盖 1D, 2D, 3D 输入
- int32 输入在函数内部/测试中转 float32 参与归一化计算（RMSNorm 本质是浮点运算）

---

## 6. 简要结论

1. Transformers 当前 RMS 归一化的绝对主流是 **标准 Llama/T5 风格**。
2. **Gemma 的 `1 + weight`** 是重要分支变体。
3. **Gated / Unweighted** 更多是特定架构需求或工程优化路径，不是通用默认形态。
4. 若你做新模型，优先选标准型；只有在架构论文明确要求时再用 gated 或其它特殊变体。
