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
Qwen3-TTS 主模型相关组件的导出层。

模块定位:
- 这里集中导出配置类、主模型类和 Processor。
- 调用者通常不直接深入 `modeling_*.py`，而是通过这里拿到最核心的三件套:
  `Qwen3TTSConfig`、`Qwen3TTSForConditionalGeneration`、`Qwen3TTSProcessor`。

Python 视角:
- 这种 `__init__` 聚合导出模式能降低导入路径耦合，
  也方便和 Hugging Face Auto 系列注册逻辑配合。
"""

from .configuration_qwen3_tts import Qwen3TTSConfig
from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from .processing_qwen3_tts import Qwen3TTSProcessor
