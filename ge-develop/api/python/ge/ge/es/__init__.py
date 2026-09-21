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
es - eager style构图基础组件

这个模块提供了图构建基础组件的Python封装，包括：
- GraphBuilder: 图构建器对象
- TensorHolder: 张量持有者对象
- list_plugins: 列出所有已加载的插件名称
- get_plugin: 获取指定名称的插件模块

同时支持通过 entry_points 机制自动加载插件包（如 es_math、es_nn 等）。
"""
__all__ = ['GraphBuilder', 'TensorHolder', 'list_plugins', 'get_plugin']

from typing import List, Union
from types import ModuleType
from .graph_builder import GraphBuilder
from .tensor_holder import TensorHolder
from ._plugin_loader import load_all_plugins
# 加载所有已注册的插件
_plugins = load_all_plugins()
# 动态挂载插件到当前命名空间
for name, module in _plugins.items():
    globals()[name] = module
    __all__.append(name)


def list_plugins() -> List[str]:
    """
    List all loaded plugin names.
    
    Returns:
        list: List of plugin names
    """
    return list(_plugins.keys())

    
def get_plugin(plugin_name: str) -> Union[ModuleType, None]:
    """
    Get the plugin module with the specified name.
    
    Args:
        plugin_name (str): Plugin name (e.g., 'math', 'nn')
    
    Returns:
        module: Plugin module object, or None if not found
    """
    return _plugins.get(plugin_name)