# Qwen3-TTS 代码阅读指南

这个文件用于配合源码中的中文注释，帮助你从“项目结构、AI 架构、Python 实现”三个层面理解仓库。

## 1. 仓库分层

### 1.1 对外入口层

- `qwen_tts/__init__.py`
  对外暴露 `Qwen3TTSModel` 和 `Qwen3TTSTokenizer`。
- `qwen_tts/__main__.py`
  `python -m qwen_tts` 的最小入口。
- `qwen_tts/cli/demo.py`
  Gradio Web UI Demo。

### 1.2 推理包装层

- `qwen_tts/inference/qwen3_tts_model.py`
  面向用户的高层 TTS 推理接口。
- `qwen_tts/inference/qwen3_tts_tokenizer.py`
  面向用户的 speech tokenizer 编解码接口。

这一层最重要的事情不是“神经网络计算”，而是:

- 统一处理路径、URL、base64、numpy 输入。
- 负责 batch 对齐、采样率对齐、device/dtype 对齐。
- 把底层模型接口整理成易用的 `from_pretrained / generate / encode / decode` 风格。

### 1.3 主模型层

- `qwen_tts/core/models/configuration_qwen3_tts.py`
  主模型配置树。
- `qwen_tts/core/models/processing_qwen3_tts.py`
  文本 processor。
- `qwen_tts/core/models/modeling_qwen3_tts.py`
  主 TTS 模型实现。

这里是整个仓库最关键的部分。

### 1.4 语音 tokenizer 层

- `qwen_tts/core/tokenizer_12hz/*`
  12Hz tokenizer。Qwen3-TTS 主模型默认依赖这一路线。
- `qwen_tts/core/tokenizer_25hz/*`
  25Hz tokenizer。更接近经典模块化声码器设计。

### 1.5 微调层

- `finetuning/prepare_data.py`
  离线提取 `audio_codes`。
- `finetuning/dataset.py`
  训练样本和 batch 打包逻辑。
- `finetuning/sft_12hz.py`
  从 Base 模型微调出自定义 speaker。

### 1.6 示例层

- `examples/*.py`
  最小可运行示例，适合先验证环境再回头读源码。

## 2. 推荐阅读顺序

如果你想快速建立全局理解，建议按下面顺序读:

1. `README.md`
2. `CODE_READING_GUIDE_ZH.md`
3. `qwen_tts/inference/qwen3_tts_model.py`
4. `qwen_tts/core/models/modeling_qwen3_tts.py`
5. `qwen_tts/inference/qwen3_tts_tokenizer.py`
6. `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`
7. `finetuning/dataset.py`
8. `finetuning/sft_12hz.py`

原因是:

- 先看高层接口，可以先知道“这个项目对外到底怎么用”。
- 再看底层建模，才能把“为什么这样调用”与“模型内部怎么做”对应起来。
- 最后看微调，会更容易理解训练数据为什么要那样排布。

## 3. 核心 AI 数据流

### 3.1 文本到语音

主链路可以概括为:

`text`
-> `processor/tokenizer`
-> `talker`
-> `codec-0`
-> `code predictor`
-> `剩余 codec 组`
-> `speech tokenizer decoder`
-> `waveform`

关键思想:

- 主模型并不直接生成波形，而是先生成离散语音 token。
- 这样做本质上是把复杂的波形建模问题，拆成“离散语音规划”和“波形重建”两部分。

### 3.2 参考音频到 voice clone 条件

对于 Base 模型，参考音频会走两条路径:

1. `ref_audio -> speech tokenizer.encode -> ref_code`
2. `ref_audio -> speaker_encoder -> ref_spk_embedding`

这两个条件分别对应:

- `ref_code`: 局部发音模式、韵律、说话方式。
- `ref_spk_embedding`: 全局音色身份。

### 3.3 多码本 codec 的生成

Qwen3-TTS 不是一次只生成一个标量 token，而是面对多个 codec 组。

设计上分成两步:

1. `talker` 先生成主 codec 通道。
2. `code predictor` 再根据主通道隐藏状态补齐其他 codec 组。

这样做的优点:

- 降低主模型的搜索难度。
- 保留多码本语音表示的重建质量。

## 4. 主模型文件怎么读

`qwen_tts/core/models/modeling_qwen3_tts.py` 很长，建议按下面逻辑看:

### 4.1 先看组件职责

- `Qwen3TTSSpeakerEncoder`
  提取说话人向量。
- `Qwen3TTSTalkerModel`
  生成主 codec 通道隐藏状态。
- `Qwen3TTSTalkerCodePredictorModel`
  补齐剩余 codec 组。
- `Qwen3TTSForConditionalGeneration`
  把所有部件装成一个可 `generate` 的完整模型。

### 4.2 再看注意力和位置编码

你会看到两套注意力:

- `Qwen3TTSTalkerAttention`
- `Qwen3TTSAttention`

原因不是“代码重复”，而是两个子网络的任务不同:

- talker 面向主序列生成；
- code predictor 面向剩余 codec 组补全。

### 4.3 最后看 `from_pretrained` 和 `generate`

这是把模型真正接成“可用产品接口”的地方。

尤其注意:

- 会额外加载 `speech_tokenizer` 子目录。
- 会读取 `generation_config.json`。
- 顶层模型本身依赖 speech tokenizer 完成参考音频编码与最终波形解码。

## 5. 12Hz 与 25Hz tokenizer 的差异

### 5.1 12Hz

特点:

- token 更稀疏，时间步更少。
- 更适合被语言模型直接建模。
- 在主模型链路里更重要。

阅读重点:

- `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`
- 看编码器如何产出多码本 token。
- 看解码器如何通过 RVQ + 上采样恢复波形。

### 5.2 25Hz

特点:

- 更偏传统声码器模块化路线。
- 需要 `xvector` 和 `reference mel` 等额外条件。
- 更适合理解“经典 TTS 模块组合”。

阅读重点:

- `qwen_tts/core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py`
- `qwen_tts/core/tokenizer_25hz/vq/*`

## 6. 微调代码怎么理解

### 6.1 `prepare_data.py`

把原始音频先变成 `audio_codes`，避免训练阶段重复跑 tokenizer。

### 6.2 `dataset.py`

这是最值得认真看的训练文件之一，因为它定义了“模型在训练时到底看到什么”。

它做了几件关键事:

- 把文本包成 chat prompt。
- 把 text ids 和 codec ids 放到同一时间轴的双通道 `input_ids` 中。
- 预留 speaker embedding 插槽。
- 构造 attention mask、embedding mask、loss mask。

### 6.3 `sft_12hz.py`

核心逻辑:

- 从 `ref_mels` 提取 speaker embedding。
- 把该向量写入输入序列的保留位置。
- 用主 loss 学 codec-0。
- 用 `sub_talker_loss` 学剩余码本组。
- 最后把目标 speaker 向量写回模型权重中一个新的 speaker id。

## 7. 你需要特别注意的 Python 语法和框架模式

### 7.1 `@dataclass`

用于“数据容器”对象，例如 `VoiceClonePromptItem`。

优点:

- 减少样板代码。
- 字段语义比裸字典更清晰。

### 7.2 `PretrainedConfig / PreTrainedModel`

这是 Hugging Face 的核心抽象。

意义:

- config 负责描述结构。
- model 负责描述计算。
- 两者共同支持 `from_pretrained()`、`save_pretrained()`。

### 7.3 `GenerationMixin`

只要模型类遵循 HF 约定并继承这个 mixin，就能直接用 `generate()`。

### 7.4 `ProcessorMixin`

用于把 tokenizer、模板等输入预处理逻辑包装成统一接口。

### 7.5 mask 与 broadcast

这个项目里大量用到:

- `attention_mask`
- `text_embedding_mask`
- `codec_embedding_mask`
- `codec_mask`

理解这些 mask 的最好办法不是死记名字，而是先回答:

- 哪些位置应该参与注意力？
- 哪些位置应该参与 embedding？
- 哪些位置应该参与 loss？

## 8. 如果你要继续深入

建议进一步做这三件事:

1. 从 `examples/` 跑通一个最小推理样例。
2. 在 `finetuning/dataset.py` 打印每个张量的 shape。
3. 跟踪一次 `generate_voice_clone()` 到 `Qwen3TTSForConditionalGeneration.generate()` 的调用链。

这样你会把“接口层、模型层、数据层”三者真正串起来。
