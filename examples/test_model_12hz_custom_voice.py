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
Custom Voice 模型示例。

阅读价值:
- 展示固定 speaker + 可选 instruct 的典型调用方式。
- 适合作为最小推理样例，而不是训练或架构阅读入口。
"""

import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    """运行一条 custom voice 推理并将结果写到本地 wav。"""
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/"

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # -------- Single (with instruct) --------
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice(
        text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        language="Chinese",
        speaker="Vivian",
        instruct="用特别愤怒的语气说",
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_custom_single.wav", wavs[0], sr)

    # -------- Batch (some empty instruct) --------
    texts = ["其实我真的有发现，我是一个特别善于观察别人情绪的人。", "She said she would be here by noon."]
    languages = ["Chinese", "English"]
    speakers = ["Vivian", "Ryan"]
    instructs = ["", "Very happy."]

    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        sf.write(f"qwen3_tts_test_custom_batch_{i}.wav", w, sr)


if __name__ == "__main__":
    main()
