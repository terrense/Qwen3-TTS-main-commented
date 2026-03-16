# Qwen3-TTS Tokenizer 报告：理论与源码对照版

这份报告的目标，不是再重复一遍论文宣传语，而是把你给出的 tokenizer 理论，与当前仓库里的实际源码结构一一对应起来，让你在看代码时能形成更具体的“模块图像”。

## 1. 先给结论

如果只看大方向，Qwen3-TTS 的 tokenizer 可以概括成两条路线：

- `25Hz tokenizer` 更像“传统高保真语音 tokenizer / 声码器系统”。
  它的离散码是单码本，编码端额外提取说话人相关条件，解码端依赖 `DiT + BigVGAN`，更偏高保真重建。
- `12Hz tokenizer` 更像“给大语言模型直接建模语音离散 token 而设计的表示系统”。
  它是 `16` 层多码本 RVQ，第一层更偏语义，后 15 层更偏声学细节，解码器是纯因果、轻量、低延迟路线。

如果只看当前开源代码的“可直接验证部分”，最关键的观察有四个：

1. 25Hz 路线的 `encode()` 会返回 `audio_codes + xvectors + ref_mels`，说明它不是“只靠离散码就能完成解码”的纯 token 体系。
2. 12Hz 路线的 `encode()` 只返回 `audio_codes`，而 `decode()` 也只需要 `audio_codes`，说明它更接近自洽的离散语音表示。
3. 12Hz 路线在结构上明确把“第 1 个码本”和“后续 15 个码本”拆开处理，这与“语义-声学解耦量化”的理论高度一致。
4. 25Hz 路线的流式 / 局部感受野特征体现在 block-based DiT 的注意力掩码里；12Hz 路线的低延迟特征体现在纯因果卷积、局部注意力和 `chunked_decode()` 的实现里。

## 2. 统一入口：源码里 tokenizer 是怎么被封装起来的

先看统一包装层 `qwen_tts/inference/qwen3_tts_tokenizer.py`。

这个文件的职责是：

- 根据 checkpoint 配置自动加载 25Hz 或 12Hz tokenizer。
- 把路径、URL、base64、numpy 波形统一转成模型输入。
- 对外提供统一的 `encode()` / `decode()` / `get_*_sample_rate()` 接口。

从这个包装层，你可以直接看出两代 tokenizer 的接口差异：

- 25Hz:
  `encode()` 输出 `audio_codes`、`xvectors`、`ref_mels`
- 12Hz:
  `encode()` 输出 `audio_codes`

这件事非常重要，因为它不是“接口设计的偶然差异”，而是两类 tokenizer 理论思路不同的直接证据：

- 25Hz 更像“离散码 + 外部声学条件共同完成重建”
- 12Hz 更像“多码本离散表示本身已经尽量包含完整重建所需的信息”

## 3. 25Hz tokenizer：理论与代码怎么对应

## 3.1 你给出的理论

你对 25Hz tokenizer 的总结可以提炼成三层：

1. 它是单码本 tokenizer，并且融合语义与声学线索。
2. 训练分两阶段：
   - 第一阶段继续预训练语音理解骨干，并加入降采样和 VQ；
   - 第二阶段再训练 / 微调 mel 解码器，把声学信息注入 token 对应的重建链路。
3. 解码端是基于 block 的流式 DiT，再接 GAN 声码器，更适合高保真，但延迟略高。

这个理论和开源代码整体是对得上的，但要注意：**当前仓库主要公开了推理结构，没有公开完整的 25Hz 两阶段训练代码**。所以你需要区分“源码能直接证实的事实”和“从论文 / README / 理论描述反推的训练背景”。

## 3.2 代码里可以直接证实什么

### 3.2.1 25Hz 的顶层模型是 `Qwen3TTSTokenizerV1Model`

位置：

- `qwen_tts/core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py`

它内部由两部分组成：

- `encoder = Qwen3TTSTokenizerV1Encoder`
- `decoder = Qwen3TTSTokenizerV1Decoder`

这说明 25Hz 路线是一个很典型的“编码器 / 解码器分离”的 tokenizer 结构。

### 3.2.2 25Hz `encode()` 不只输出离散码

在 `Qwen3TTSTokenizerV1Model.encode()` 中，代码会做三件事：

1. 调 `self.encoder.quantize_speech(wavs)` 得到离散码 `codes`
2. 对每条 wav 调 `self.encoder_xvector_extractor.extract_code(...)` 得到 `xvector`
3. 同时得到 `ref_mel`

最后返回：

- `codes`
- `xvectors`
- `ref_mels`

这说明 25Hz 的重建并不是“只靠 token 自己”完成的，而是显式依赖额外条件。

### 3.2.3 25Hz 当前推理接口表现为“单码流”

从推理包装层和 V1 模型的 `encode()/decode()` 形状可以看出来：

- 25Hz 的 `audio_codes` 形状更像 `(codes_len,)`
- 12Hz 的 `audio_codes` 形状是 `(codes_len, num_quantizers)`

也就是说，25Hz 在当前开源实现中的对外离散序列是单主码流，而不是 16 层多码本这种结构。

再往里看 `qwen_tts/core/tokenizer_25hz/vq/speech_vq.py`：

- `WhisperEncoderVQ` 内部创建的是 `DistributedGroupResidualVectorQuantization`
- 但初始化时用了：
  - `num_groups=1`
  - `num_quantizers=1`

这进一步支持了“当前发布实现中，25Hz 路线是单码本 / 单主离散流”的判断。

### 3.2.4 25Hz 编码端不是纯随机 VQ，而是“语音骨干 + 量化”

`WhisperEncoderVQ` 继承自 `WhisperEncoder`。

在结构上它做的是：

`log-mel -> conv -> transformer encoder -> VQ`

这和你说的“先有语音理解骨干，再加降采样和 VQ”是同一个方向。

需要注意的是：

- 当前代码里骨干名叫 `WhisperEncoder`
- 不是显式写成 `Qwen2-Audio`

所以更准确的说法应该是：

- **从理论上，你可以把它理解为一种 ASR / 语音理解骨干上接 VQ 的路线**
- **但当前开源代码层面，直接可见的是 Whisper 风格编码器 + 量化器，而不是名义上公开的 Qwen2-Audio 训练骨干**

### 3.2.5 25Hz 解码端确实是 `DiT + BigVGAN`

`Qwen3TTSTokenizerV1Decoder` 的前向逻辑非常清楚：

1. `self.dit.sample(...)` 先生成 mel spectrogram
2. `self.bigvgan(mel_spectrogram)` 再把 mel 转成 waveform

这和你说的：

- block-based 流式 DiT
- 改进 GAN 声码器重建波形

是高度一致的。

### 3.2.6 block-based 流式注意力在代码里有具体实现

`Qwen3TTSTokenizerV1DecoderDiTConfig` 中有这几个很关键的配置：

- `block_size=24`
- `look_ahead_layers=[10]`
- `look_backward_layers=[0, 20]`

在 `Qwen3TTSTokenizerV1DecoderDiTModel` 里：

- `_create_block_diff()` 会先按 `block_size` 计算每个 token 属于哪个 block
- `DiTDecoderLayer.forward()` 里会根据 `block_diff` 构建块级注意力掩码

注意力掩码条件是：

- 允许看有限的前向 block
- 允许看有限的后向 block

这就是你说的“基于 Block 的流式 DiT / 滑动窗口注意力”的代码化表达。

不过这里有一个细节要讲清楚：

- 你给出的理论说“感受野为 4 个 Block”
- 当前开源配置里直接可见的是：
  - 某些层 `look_ahead_block=1`
  - 某些层 `look_backward_block=1`

所以更严谨的表达应该是：

- **代码明确体现了 block 级局部注意力思想**
- **但“4 个 Block 感受野”这个总表述，更像是论文 / 系统设计层的整体描述，不一定能直接从这几个默认配置值一眼等价推出**

## 3.3 25Hz 路线该怎么建立直觉

你可以把 25Hz tokenizer 想成：

“先把语音压成一个离散主索引，再用外部说话人向量和参考 mel 帮 decoder 做高保真重建。”

它像什么？

- 像一个“离散化过的高质量条件声码器系统”
- 不像一个“仅靠 token 就完整自洽”的纯离散语言建模语音系统

所以：

- 它的强项是高保真、条件充分、重建质量强
- 它的代价是解码链更重，外部条件更多，延迟更难做到极低

## 4. 12Hz tokenizer：理论与代码怎么对应

## 4.1 你给出的理论

你对 12Hz tokenizer 的总结很清晰：

1. 它是 16 层多码本 RVQ tokenizer。
2. 第一层偏语义，后 15 层偏声学细节 / 韵律。
3. 第一层语义码本训练时会和 WavLM teacher 的语义表征对齐。
4. 训练是 GAN 框架，目标是低比特率、高自然度、超低延迟。
5. 解码器是纯左上下文因果结构，不依赖 x-vector、不依赖复杂扩散模型。

这套理论和当前开源代码的结构匹配度非常高，尤其是“第一层语义、后续 15 层声学”的设计，在源码里是直接可见的。

## 4.2 代码里可以直接证实什么

### 4.2.1 12Hz 顶层模型是 `Qwen3TTSTokenizerV2Model`

位置：

- `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`

它由两部分组成：

- `encoder = Qwen3TTSTokenizerV2Encoder`
- `decoder = Qwen3TTSTokenizerV2Decoder`

和 25Hz 一样也是 encoder / decoder 结构，但内部设计明显更偏“统一离散建模”。

### 4.2.2 12Hz 是 16 层多码本

在 `Qwen3TTSTokenizerV2DecoderConfig` 里，默认值直接写着：

- `num_quantizers=16`
- `codebook_size=2048`

在 `Qwen3TTSTokenizerV2Config` 里还有：

- `encoder_valid_num_quantizers=16`

这两处共同说明：

- 当前开源 12Hz tokenizer 确实就是 16 层多码本
- 并且 encoder 输出时也只保留 16 个有效量化层

### 4.2.3 “第一层语义、后续 15 层声学”在代码里是显式设计

这一点是本报告里最能帮助你把理论和代码对上的地方。

看 `SplitResidualVectorQuantizer`：

- `n_q_semantic=1`
- `n_q_acoustic = n_q - n_q_semantic`

解码逻辑是：

1. 先用 `rvq_first.decode(codes[:, :1])`
2. 再把 `rvq_rest.decode(codes[:, 1:])` 加上去

这几乎就是把你的理论直接翻译成代码：

- 第 1 个码本 = semantic quantizer
- 后面 15 个码本 = acoustic quantizers

也就是说，**“语义-声学解耦量化”不是你从论文里抽象出来的概念，而是源码里真实存在的结构划分。**

### 4.2.4 12Hz 解码时不需要 x-vector / ref_mel

`Qwen3TTSTokenizerV2Model.decode()` 只接收：

- `audio_codes`

然后直接：

- `self.decoder.chunked_decode(audio_codes.transpose(1, 2))`

这意味着：

- 12Hz tokenizer 的波形重建链是“纯 token 驱动”的
- 它不像 25Hz 那样还需要说话人向量和参考 mel 作为额外解码条件

这也是为什么 12Hz 更适合作为主 TTS 模型统一建模的基础表示。

### 4.2.5 12Hz 解码器确实是“轻量、因果、可分块”的

`Qwen3TTSTokenizerV2Decoder` 的结构可以概括成：

`codes -> RVQ decode -> causal conv -> transformer -> 多级上采样 -> 因果卷积波形头`

具体体现为：

- `pre_conv = Qwen3TTSTokenizerV2CausalConvNet`
- `pre_transformer = Qwen3TTSTokenizerV2DecoderTransformerModel`
- 上采样模块使用 `Qwen3TTSTokenizerV2CausalTransConvNet`
- 最终波形头仍然是因果卷积

并且：

- `chunked_decode()` 支持按块解码长序列
- `sliding_window=72`
- 多个卷积模块都写成了 causal 版本

这和你说的“纯左上下文、无 look-ahead 需求、token 一生成即可立即解码”的设计目标非常一致。

### 4.2.6 12Hz 的“12Hz”在代码里更准确地表现为 12.5Hz

这是一个很值得你建立直觉的小细节。

在 `Qwen3TTSTokenizerV2Config` 中：

- `input_sample_rate=24000`
- `encode_downsample_rate=1920`
- `decode_upsample_rate=1920`

这意味着每个离散时间步对应：

- `1920 / 24000 = 0.08s`

也就是：

- `1 / 0.08 = 12.5 fps`

README 的 benchmark 表里也明确写的是 `12.5`。

所以你可以这样理解：

- “Qwen3-TTS-Tokenizer-12Hz” 是产品命名 / 近似命名
- 从当前开源配置和 README benchmark 来看，实际步率是 `12.5Hz`

## 4.3 代码不能直接证实，但和理论高度一致的部分

这里有两点要特别区分：

### 4.3.1 WavLM teacher 蒸馏

你的理论里提到：

- 第一层语义码本会和 WavLM teacher 对齐

这个说法从结构上是合理的，也和 `n_q_semantic=1` 很契合。

但当前开源仓库中的推理代码里：

- 没有直接出现 `WavLM`
- 没有公开第一层语义蒸馏的训练损失实现

所以更严谨的说法是：

- **源码保留了“第一层语义码本”的架构痕迹**
- **但 teacher 蒸馏过程属于训练时细节，当前发布仓库没有完整公开**

### 4.3.2 GAN 训练细节

你提到：

- 生成器直接处理原始波形提取量化特征
- 判别器提升自然度
- 多尺度 mel 重建损失保证时频一致性

这类信息更像训练框架设计。

而当前仓库里公开的是：

- 推理端 encoder / decoder 结构
- 不是完整 GAN 训练脚本

所以这里同样要区分：

- **结构方向与理论是一致的**
- **但完整 adversarial loss / discriminator 代码没有在这个开源推理仓库中完整展开**

## 5. 25Hz 与 12Hz 的核心对比

| 维度 | 25Hz tokenizer | 12Hz tokenizer |
|---|---|---|
| 对外离散形式 | 单主码流 | 16 层多码本 |
| 解码条件 | `codes + xvector + ref_mel` | `codes` 即可 |
| 重建链路 | `DiT -> BigVGAN` | `RVQ decode -> causal conv/transformer -> waveform` |
| 理论侧重点 | 高保真重建、条件丰富 | 极低码率、极低延迟、适合主模型直接建模 |
| 流式特征 | block-based 局部注意力 | 纯因果、chunked decode、无复杂外部条件 |
| 和主 TTS 模型的匹配度 | 更像外部高质量声码器体系 | 更像主模型原生离散语音接口 |

## 6. 你学习时最应该抓住的“具象化观察”

如果你想把理论真正看成代码里的具体东西，最推荐你盯住下面 5 个观察点。

### 6.1 看 `encode()` 输出长什么样

这是最直观的：

- 25Hz：输出三样东西
- 12Hz：输出一个多码本 token 张量

你一旦抓住这一点，两个 tokenizer 的哲学差异会立刻变具体。

### 6.2 看 `decode()` 需要什么条件

- 25Hz 需要 `xvectors` 和 `ref_mels`
- 12Hz 不需要

这直接决定了：

- 谁更像条件声码器
- 谁更像自洽离散表示

### 6.3 看 12Hz 的 `SplitResidualVectorQuantizer`

这是把“第一层语义、后续声学”写成代码的核心类。

如果你只打算精读一个 12Hz 细节，就读它。

### 6.4 看 25Hz 的 `Qwen3TTSTokenizerV1Decoder`

因为它把整个高保真重建链写得最清楚：

- 先 DiT 生成 mel
- 再 BigVGAN 出波形

这能帮你立刻建立“为什么 25Hz 路线重，但音质强”的直觉。

### 6.5 看 12Hz 的 `chunked_decode()`

这个函数很短，但价值很高。

它直接把“长序列按块解码、保留左上下文、裁掉重复上下文”的低延迟思路落成了代码。

## 7. 一个你必须知道的边界：理论不等于仓库完整公开内容

当前这个仓库更偏：

- tokenizer 推理实现
- 主模型推理实现
- 一部分微调脚本

而不完全是：

- tokenizer 全量训练仓库
- 论文所有训练 trick 的公开复现

因此下列内容你可以在这份报告里当作“高可信理论背景”，但不要误以为已经在当前仓库中完整出现：

- 25Hz 第一阶段 ASR 继续预训练的完整训练管线
- 25Hz 第二阶段 mel 重建 / 联合任务训练细节
- 12Hz 第一层语义码本和 WavLM teacher 对齐的具体 loss
- 12Hz GAN 判别器与多尺度 mel loss 的完整训练代码

更准确的说法是：

- **架构指纹已经公开**
- **训练细节并未完整公开**

## 8. 最后给你一个学习路线

如果你想把这套 tokenizer 真正学透，我建议按这个顺序：

1. 先读 `qwen_tts/inference/qwen3_tts_tokenizer.py`
   先建立统一接口直觉。
2. 再读 25Hz 顶层：
   `Qwen3TTSTokenizerV1Model`
3. 再读 25Hz 编码 / 解码关键部件：
   `WhisperEncoderVQ`、`XVectorExtractor`、`Qwen3TTSTokenizerV1Decoder`
4. 再读 12Hz 顶层：
   `Qwen3TTSTokenizerV2Model`
5. 最后精读：
   `SplitResidualVectorQuantizer`、`Qwen3TTSTokenizerV2Decoder`

你可以把这条路线理解成：

- 先看“接口差异”
- 再看“25Hz 为什么重”
- 再看“12Hz 为什么轻”
- 最后把“语义码本 / 声学码本 / 低延迟解码”串起来

## 9. 一句话总结

25Hz tokenizer 更像“高保真条件重建系统”，12Hz tokenizer 更像“为大模型直接建模语音 token 而生的低比特率离散表示系统”；前者重条件、重音质，后者重表示、重延迟，而当前 Qwen3-TTS 主模型明显是更偏向 12Hz 这条路线来完成统一建模的。
