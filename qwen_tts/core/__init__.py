# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
核心语音 tokenizer 组件的导出层。

模块定位:
- `qwen_tts.core` 不直接暴露最终的 TTS 生成模型，而是聚合两代语音 tokenizer:
  25Hz 的 V1 和 12Hz 的 V2。

AI 视角:
- 25Hz 版本更像“经典声码器式”方案，包含 x-vector、参考 mel、DiT、BigVGAN 等部件。
- 12Hz 版本更偏向统一离散语音 token 建模，压缩率更高，也更适合作为 Qwen3-TTS
  主模型的声学离散接口。

工程视角:
- 这里的 re-export 让推理包装层只依赖 `qwen_tts.core`，不必关心内部文件层级。
"""

from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config
from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model
from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
