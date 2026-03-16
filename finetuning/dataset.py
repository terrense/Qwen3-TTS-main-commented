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
Qwen3-TTS 12Hz 微调数据集定义。

模块定位:
- 该文件把 jsonl 中的一条训练样本，整理成 Qwen3-TTS 微调所需的多通道输入张量。

AI 视角:
- 训练目标不是简单的“文本 -> codec”，而是让模型在统一序列里同时看到:
  文本 token、主 codec 通道、speaker 条件、以及其余 codec 组。
- 其中 `ref_mel` 用于 speaker encoder 提取目标说话人向量，
  `audio_codes` 则是监督信号本身。

Python 视角:
- `Dataset.__getitem__` 负责单样本解析，
  `collate_fn` 负责把变长样本打包成 batch。
- 这一层最容易出错的不是模型，而是张量形状、mask 对齐和特殊 token 插槽位置。
"""

from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    """把带离散码的 TTS 样本转成训练 batch 所需张量。

    样本字段约定:
    - `text`: 目标要合成的文本。
    - `audio_codes`: 由 tokenizer 预先提取好的离散声学码。
    - `ref_audio`: 用于提取目标说话人 mel 的参考音频。

    重要实现思想:
    - 文本和 codec 实际上共享同一条时间轴，但分别占据双通道 `input_ids[..., 0/1]`。
    - speaker embedding 不通过额外字段注入，而是占用 codec 通道中的一个预留位置，
      这样能与原模型结构对齐。
    """
    def __init__(self, data_list, processor, config:Qwen3TTSConfig, lag_num = -1):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        """从本地路径读取音频并转为单声道 `float32` 波形。"""
        audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    
    def _build_assistant_text(self, text: str) -> str:
        """构造与预训练/指令微调一致的 assistant 侧 prompt 模板。"""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        """将标量包装成列表，简化后续统一处理。"""
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = inputs["input_ids"]
        # 这里兼容 tokenizer 返回一维与二维两种情况，确保后续一律按 batch 维处理。
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        """提取 24kHz 参考 mel，供 speaker encoder 生成目标音色向量。"""
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        return mels



    def __getitem__(self, idx):
        """把一条 jsonl 样本转成模型训练所需的最小字段集合。"""
        item = self.data_list[idx]

        audio_path  = item["audio"]
        text        = item["text"]
        audio_codes = item["audio_codes"]
        language        = item.get('language','Auto')
        ref_audio_path  = item['ref_audio']

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav,sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            # 这里裁掉结尾若干模板 token，是为了让后续手工拼接的 codec 起始区间与训练布局对齐。
            "text_ids": text_ids[:,:-5],    # (1, text_len)
            "audio_codes":audio_codes,      # (codec_len, num_quantizers)
            "ref_mel":ref_mel
        }
        
    def collate_fn(self, batch):
        """把变长样本打包成双通道自回归训练输入。

        返回张量中最关键的几项:
        - `input_ids[..., 0]`: 文本通道，承载 text token 与若干 TTS 特殊 token。
        - `input_ids[..., 1]`: codec 主通道，承载 codec-0、自定义占位符和 speaker 插槽。
        - `codec_ids`: 16 个码本的完整监督标签。
        - 各类 `*_mask`: 控制哪些位置参与 embedding、attention 和 loss 计算。
        """
        assert self.lag_num == -1

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b,t = len(batch),max_length

        input_ids   = torch.zeros((b,t,2),dtype=torch.long)
        codec_ids   = torch.zeros((b,t,16),dtype=torch.long)
        text_embedding_mask     = torch.zeros((b,t),dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b,t),dtype=torch.bool)
        codec_mask      = torch.zeros((b,t),dtype=torch.bool)
        attention_mask  = torch.zeros((b,t),dtype=torch.long)
        codec_0_labels  = torch.full((b, t), -100, dtype=torch.long)

        for i,data in enumerate(batch):
            text_ids        = data['text_ids']
            audio_codec_0   = data['audio_codes'][:,0]
            audio_codecs    = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # 文本通道: 以聊天模板 token 开头，在中段插入 TTS 特殊 token，
            # 再把真实文本与后续 codec 区域占位拼接到同一时间轴上。
            input_ids[i,  :3, 0] = text_ids[0,:3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i,   7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0,3:]
            input_ids[i,   8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len , 0] = self.config.tts_pad_token_id
            text_embedding_mask[i,  :8+text_ids_len+codec_ids_len] = True

            # codec 通道:
            # - 前几个位置是“系统保留槽位”，其中索引 6 会在训练时被真实 speaker embedding 覆盖。
            # - 从 `codec_bos_id` 开始进入真正的 codec 自回归预测区。
            input_ids[i,    3:8 ,1] = torch.tensor(
                                        [
                                            self.config.talker_config.codec_nothink_id,
                                            self.config.talker_config.codec_think_bos_id,
                                            self.config.talker_config.codec_think_eos_id,
                                            0,     # for speaker embedding
                                            self.config.talker_config.codec_pad_id       
                                        ]
                                    )
            input_ids[i,    8:8+text_ids_len-3  ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-3    ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-2    ,1] = self.config.talker_config.codec_bos_id
            input_ids[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,    1] = audio_codec_0
            input_ids[i,    8+text_ids_len-1+codec_ids_len,    1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i,    8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,:] = audio_codecs

            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False       # 该位置不走离散 embedding，而是留给 speaker embedding 直接写入。

            codec_mask[i,   8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True
        
        ref_mels = [data['ref_mel'] for data in batch]
        ref_mels = torch.cat(ref_mels,dim=0)

        return {
            'input_ids':input_ids,
            'ref_mels':ref_mels,
            'attention_mask':attention_mask,
            'text_embedding_mask':text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask':codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels':codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask':codec_mask
        }
