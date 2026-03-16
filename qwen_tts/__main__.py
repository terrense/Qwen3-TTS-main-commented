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
`python -m qwen_tts` 的入口文件。

模块定位:
- 这里只提供一个非常轻量的提示信息，告诉用户应使用真正的命令行入口
  `qwen-tts-demo`。

Python 语法视角:
- `if __name__ == "__main__"` 是 Python 模块既可被导入、也可被直接执行时
  的常见分发写法。
"""

def main():
    """打印包的简要说明，避免把 Web UI 逻辑塞进 `python -m` 入口。"""
    print(
        "qwen_tts package.\n"
        "Use CLI entrypoints:\n"
        "  - qwen-tts-demo\n"
    )

if __name__ == "__main__":
    main()
