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
离线语音 tokenizer 预处理脚本。

模块定位:
- 这个脚本把原始训练 jsonl 中的音频字段转成 `audio_codes`，
  供后续 `sft_12hz.py` 直接读取。

AI 视角:
- 先离线提取离散语音码，可以避免训练时重复跑 tokenizer，
  显著降低训练阶段的前处理开销。

Python/工程视角:
- 脚本采用简单的“读入全部样本 -> 小批量 encode -> 回写 jsonl”流程，
  逻辑直白，方便二次改造成更复杂的数据流水线。
"""

import argparse
import json

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32

def main():
    """批量读取 jsonl，提取 `audio_codes` 后写回新的 jsonl 文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    total_lines = open(args.input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]

    final_lines = []
    batch_lines = []
    batch_audios = []
    for line in total_lines:
        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= BATCH_INFER_NUM:
            # 批量跑 tokenizer 能显著减少 Python 调度与 I/O 开销。
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line['audio_codes'] = code.cpu().tolist()
                final_lines.append(line)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        # 处理最后一个不足 batch size 的尾批次。
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, line in zip(enc_res.audio_codes, batch_lines):
            line['audio_codes'] = code.cpu().tolist()
            final_lines.append(line)
        batch_lines.clear()
        batch_audios.clear()

    final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

    with open(args.output_jsonl, 'w') as f:
        for line in final_lines:
            f.writelines(line + '\n')

if __name__ == "__main__":
    main()
