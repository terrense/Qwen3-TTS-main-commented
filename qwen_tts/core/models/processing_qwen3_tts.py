# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
Qwen3-TTS 的文本 Processor。

模块定位:
- 这个文件只处理“文本侧”的预处理，不处理语音特征。
- 在 Hugging Face 生态里，Processor 往往承担“把若干输入模态整理成模型可消费格式”
  的角色；这里因为语音 prompt 在模型内部另行处理，所以该 Processor 实际上是
  一个带聊天模板能力的 tokenizer 薄封装。

AI 视角:
- Qwen3-TTS 训练和推理都把文本组织成 chat-like prompt，
  因此 Processor 的关键价值不是分词本身，而是稳定地复用聊天模板与 padding 规则。

Python 视角:
- 继承 `ProcessorMixin` 能直接复用 Transformers 中统一的保存、加载、模板应用接口。
"""

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin


class Qwen3TTSProcessorKwargs(ProcessingKwargs, total=False):
    """为 Processor 提供默认关键字参数。

    这里继承 `TypedDict` 风格的 `ProcessingKwargs`，本质上是在声明
    `Processor.__call__` 支持哪些嵌套 kwargs，以及它们的默认值。
    """
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        }
    }

class Qwen3TTSProcessor(ProcessorMixin):
    r"""
    构造 Qwen3-TTS 文本处理器。

    Args:
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.

    设计说明:
    - 这里没有额外的 acoustic feature extractor，是因为语音参考信息并不走
      `processor(...)` 这一层，而是在更高层推理包装类中单独归一化并送入模型。
    - 因此这个类的主要职责是保证文本 tokenization、padding 方向、chat template
      的行为与训练时一致。
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self, tokenizer=None, chat_template=None
    ):
        super().__init__(tokenizer, chat_template=chat_template)

    def __call__(self, text=None, **kwargs) -> BatchFeature:
        """
        将文本编码为模型输入。

        这里虽然沿用了 Processor 的通用接口命名，但实际只处理文本。
        语音参考信息的加载、重采样、speaker embedding 提取都在更高层包装类中完成。

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3TTSProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if not isinstance(text, list):
            text = [text]

        # 这里把调用参数拆成 `text_kwargs`，是 Hugging Face Processor 生态的常见模式:
        # 由 `_merge_kwargs` 负责把默认参数、实例初始化参数和本次调用参数合并。
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        """兼容单条对话与 batch 对话两种输入形态。"""
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
            )
        )


__all__ = ["Qwen3TTSProcessor"]
