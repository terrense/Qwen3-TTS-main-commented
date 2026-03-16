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
Qwen3-TTS 的包级入口。

模块定位:
- 这个文件决定了 `import qwen_tts` 时，包对外暴露哪些高层抽象。
- 当前仓库主要向外公开两个对象:
  `Qwen3TTSModel` 负责文本到语音生成，
  `Qwen3TTSTokenizer` 负责音频到离散码的编解码。

AI 视角:
- `Qwen3TTSModel` 对应“文本/提示 -> 语音”的主生成链路。
- `Qwen3TTSTokenizer` 对应“波形 <-> 离散声学 token”的声码器/语音 tokenizer 链路。

Python 视角:
- `__init__.py` 经常用来做 re-export，目的是把深层目录里的类提升到包顶层，
  让调用方可以写出更稳定、简洁的导入语句。
"""

from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = ["__version__"]
