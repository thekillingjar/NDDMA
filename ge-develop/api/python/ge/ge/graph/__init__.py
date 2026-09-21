#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
graph模块 - 图操作接口

这个模块提供了对Graph Engine核心图操作功能的Python封装，包括：
- Graph: 图对象
- Node: 节点对象
- Tensor: 张量对象
- DataType: 数据类型
- Format: 数据格式
- DumpFormat:  dump格式
"""

from .graph import Graph, DumpFormat
from .node import Node
from .tensor import Tensor
from .types import DataType, Format

__all__ = [
    'Graph',
    'Node',
    'DataType',
    'Format',
    'Tensor',
    'DumpFormat',
]
