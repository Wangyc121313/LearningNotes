# VLM — Vision-Language Model (视觉语言模型)

> 多模态大模型学习笔记：在 LLM 基础上引入视觉感知能力

## 与 LLM / VLA 的关系

```
LLM  ──────────►  VLM  ──────────►  VLA
（文本）         （文本 + 视觉）    （文本 + 视觉 + 动作）
```

## 内容概览

- 多模态架构（Vision Encoder + LLM Decoder）
- 代表模型：CLIP、LLaVA、InternVL、Qwen-VL、GPT-4V
- 视觉问答（VQA）、图像描述（Image Captioning）
- 指令微调（Instruction Tuning）与对齐

## 1. VLM 架构详解

VLM（Vision-Language Model）= 视觉编码器 + 连接模块 + 语言模型

```
图像输入 (H×W×3)
      │
      ▼
 Vision Encoder（ViT / CLIP Image Encoder）
      │  视觉特征 [N_patches × D_vision]
      ▼
 Visual Projection / Resampler
 （将视觉特征映射到语言空间）
      │  视觉 token [N' × D_lm]
      ▼
 Language Model（LLaMA / Qwen / Mistral ...）
      │  交错处理 [视觉 token + 文本 token]
      ▼
 生成文本输出（描述/回答/分析...）
```

### 视觉-语言连接方式

| 方式 | 代表 | 说明 |
|------|------|------|
| **Linear Projection** | LLaVA-1 | 最简单，直接线性映射 |
| **MLP Projection** | LLaVA-1.5 | 2层 MLP，效果更好 |
| **Q-Former** | BLIP-2 | 用可学习 query 做交叉注意力压缩 |
| **Resampler** | Flamingo/InternVL2 | 固定数量的可学习 query token |
| **Cross Attention** | Flamingo | 视觉特征通过 cross-attn 注入每层 LM |

---

## 2. CLIP：视觉语言对比学习基础

CLIP（Contrastive Language-Image Pretraining，OpenAI 2021）是大多数 VLM 的视觉骨干。

**训练**：在 4 亿图文对上，使图文 embedding 余弦相似度最大化（正对），最小化（负对）：

$$L = -\frac{1}{N}\sum_{i}\log\frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_j\exp(\text{sim}(v_i, t_j)/\tau)}$$

**能力**：Zero-shot 分类（无需任务特定训练）、图文检索、图像特征提取

---

## 3. 代表模型一览

| 模型 | 机构 | 视觉编码器 | LM | 特点 |
|------|------|---------|-----|------|
| **BLIP-2** | Salesforce | ViT-G + Q-Former | FlanT5 / Vicuna | Q-Former 高效连接 |
| **LLaVA-1.5** | Wisconsin/MS | CLIP ViT-L | LLaMA/Vicuna | 简单有效，开源 |
| **InternVL2** | Shanghai AI Lab | InternViT | InternLM | 超强开源，媲美 GPT-4V |
| **Qwen2-VL** | Alibaba | ViT + RoPE | Qwen2 | 支持动态分辨率、视频 |
| **LLaMA-3.2-Vision** | Meta | ViT | LLaMA 3.2 | 开源，多模态推理强 |
| **GPT-4V/4o** | OpenAI | 未公开 | GPT-4 | 商业最强 |
| **Gemini** | Google | 未公开 | Gemini | 原生多模态 |
| **CLIP** | OpenAI | ResNet/ViT | Transformer | 图文对齐基础模型 |

---

## 4. 主要任务

| 任务 | 说明 | 示例 |
|------|------|------|
| **VQA（视觉问答）** | 根据图像回答问题 | "图中有几只猫？" |
| **Image Captioning** | 描述图像内容 | 生成图像的文字描述 |
| **OCR / 文档理解** | 识别图中文字/表格 | 识别发票、扫描件 |
| **视觉推理** | 图像内容的逻辑推理 | "哪个物体更大？" |
| **图文检索** | 以图搜图 / 以文搜图 | CLIP 零样本检索 |
| **视频理解** | 视频内容理解和问答 | 长视频摘要 |
| **GUI/Agent** | 屏幕截图理解，自动操作 | 网页自动化 |

---

## 5. 指令微调与对齐

VLM 预训练后通常需要 SFT（监督微调）使其遵循指令：

```
训练数据格式（LLaVA 风格）：
{
  "image": "cat.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n这张图中有什么动物？"},
    {"from": "gpt",   "value": "图中有一只橙色的猫咪，正趴在沙发上。"}
  ]
}
```

**LoRA 微调**：只训练少量适配器参数（约 1-5% 参数），大幅降低显存和时间需求。

```python
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def load_sample_image(url=None):
    """加载示例图像"""
    if url:
        try:
            resp = requests.get(url, timeout=5)
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            pass
    # 生成随机彩色图像
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)

# ===== 1. CLIP：图文相似度计算 =====
try:
    from transformers import CLIPProcessor, CLIPModel

    print("加载 CLIP 模型 (openai/clip-vit-base-patch32)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    image = load_sample_image()
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a sunset over the ocean",
        "a red apple on a table",
        "people walking in a city",
    ]

    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)

    # 图文相似度（logits_per_image 已做温度缩放）
    probs = outputs.logits_per_image.softmax(dim=1).squeeze()
    print("\n[CLIP 零样本分类 - 各文本与图像的相似度]")
    for text, prob in sorted(zip(texts, probs.tolist()), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {prob:.3f} {bar:<30} {text}")

    # 图像特征提取（用于图文检索）
    image_features = clip_model.get_image_features(
        pixel_values=inputs["pixel_values"]
    )
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = clip_model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    cosine_sims = (image_features @ text_features.T).squeeze()
    print(f"\n[CLIP embedding 维度] 图像: {image_features.shape}, 文本: {text_features.shape}")
    print(f"余弦相似度: {cosine_sims.tolist()}")

except ImportError:
    print("安装 transformers: pip install transformers")

# ===== 2. LLaVA 风格 VLM 推理 =====
try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    print("\n\n加载 LLaVA-Next 模型 (llava-hf/llava-v1.6-mistral-7b-hf)...")
    print("注意：需要约 16GB 显存，首次下载较慢")

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    image = load_sample_image()
    # LLaVA 对话格式
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "请详细描述这张图片中的内容。"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    generated = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
    print(f"LLaVA 回答: {generated}")

except (ImportError, Exception) as e:
    print(f"\nLLaVA 示例跳过: {e}")
    print("完整示例需要: pip install transformers accelerate + GPU 16GB+")

# ===== 3. Qwen2-VL 推理（更推荐：开源中最强之一）=====
print("\n===== Qwen2-VL 推理示例 =====")
qwen_vl_code = """
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 加载模型（2B 版本约 4GB 显存）
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 支持图像 URL / 本地路径 / PIL Image / Base64
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://example.com/image.jpg"},
            {"type": "text", "text": "图中有什么？请详细描述。"}
        ]
    }
]

# 处理输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to("cuda")

# 推理
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=512)
output = processor.decode(output_ids[0][inputs.input_ids.shape[1]:],
                           skip_special_tokens=True)
print("Qwen2-VL:", output)
"""
print(qwen_vl_code)

# ===== 4. 用 vLLM 部署 VLM =====
print("===== 用 vLLM 高效部署 VLM =====")
vllm_vlm_code = """
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import encode_image_base64
import base64

# vLLM 支持 VLM 推理（同样有 PagedAttention 优化）
llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    max_model_len=8192,
    limit_mm_per_prompt={"image": 5},   # 每次请求最多5张图
)

params = SamplingParams(temperature=0.7, max_tokens=512)

# 读取图像 → Base64
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

outputs = llm.generate({
    "prompt": "<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>请描述这张图片<|im_end|>\\n<|im_start|>assistant\\n",
    "multi_modal_data": {"image": f"data:image/jpeg;base64,{img_b64}"}
}, params)
print(outputs[0].outputs[0].text)
"""
print(vllm_vlm_code)
```
