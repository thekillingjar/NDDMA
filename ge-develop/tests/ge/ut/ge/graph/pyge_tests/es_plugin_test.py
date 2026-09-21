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
ES 插件功能测试 - 使用 pytest 框架
测试 ge.es 模块中的插件加载和管理功能
"""

import pytest
import sys
import os
import types
from unittest.mock import Mock, patch, MagicMock

# 需要添加 ge 到 Python 路径，否则会报错
try:
    from ge.es import list_plugins, get_plugin
    from ge.es._plugin_loader import (
        load_all_plugins,
        _coerce_to_module,
        _iter_plugin_entry_points
    )
except ImportError as e:
    pytest.skip(f"无法导入 ge 模块: {e}", allow_module_level=True)


class TestESPlugins:
    """ES 插件功能测试类"""

    def test_list_plugins(self):
        """测试 list_plugins() 函数"""
        plugins = list_plugins()
        assert isinstance(plugins, list)

    def test_get_plugin_existing(self):
        """测试 get_plugin() 获取已存在的插件"""
        plugins = list_plugins()
        if plugins:
            plugin = get_plugin(plugins[0])
            assert plugin is not None
            assert isinstance(plugin, types.ModuleType)

    def test_get_plugin_nonexistent(self):
        """测试 get_plugin() 获取不存在的插件"""
        plugin = get_plugin("nonexistent_plugin_xyz")
        assert plugin is None

    def test_get_plugin_with_none(self):
        """测试 get_plugin() 使用 None 作为参数"""
        plugin = get_plugin(None)
        assert plugin is None


class TestPluginLoader:
    """插件加载器功能测试类"""

    def test_coerce_to_module_with_module(self):
        """测试 _coerce_to_module() 传入模块对象"""
        mock_module = types.ModuleType("test_module")
        result = _coerce_to_module(mock_module, "test")
        assert result is mock_module

    def test_coerce_to_module_with_string(self):
        """测试 _coerce_to_module() 传入字符串"""
        with patch('importlib.import_module') as mock_import:
            mock_module = types.ModuleType("test_module")
            mock_import.return_value = mock_module
            result = _coerce_to_module("test.module", "test")
            assert result is mock_module
            mock_import.assert_called_once_with("test.module")

    def test_coerce_to_module_with_callable(self):
        """测试 _coerce_to_module() 传入可调用对象"""
        mock_module = types.ModuleType("test_module")
        callable_obj = lambda: mock_module
        result = _coerce_to_module(callable_obj, "test")
        assert result is mock_module

    def test_coerce_to_module_with_callable_returning_string(self):
        """测试 _coerce_to_module() 传入返回字符串的可调用对象"""
        with patch('importlib.import_module') as mock_import:
            mock_module = types.ModuleType("test_module")
            mock_import.return_value = mock_module
            callable_obj = lambda: "test.module"
            result = _coerce_to_module(callable_obj, "test")
            assert result is mock_module
            mock_import.assert_called_once_with("test.module")

    def test_coerce_to_module_invalid_type(self):
        """测试 _coerce_to_module() 传入无效类型"""
        with pytest.raises(TypeError):
            _coerce_to_module(123, "test")

    @patch('ge.es._plugin_loader.entry_points')
    def test_iter_plugin_entry_points_python310_plus(self, mock_entry_points):
        """测试 _iter_plugin_entry_points() Python 3.10+ 路径"""
        mock_ep = Mock()
        mock_entry_points.return_value = [mock_ep]
        result = list(_iter_plugin_entry_points())
        assert len(result) == 1
        assert result[0] is mock_ep

    @patch('ge.es._plugin_loader.entry_points')
    def test_iter_plugin_entry_points_dict_format(self, mock_entry_points):
        """测试 _iter_plugin_entry_points() 字典格式返回"""
        mock_ep = Mock()
        mock_entry_points.return_value = {"ge.es.plugins": [mock_ep]}
        result = list(_iter_plugin_entry_points())
        assert len(result) == 1
        assert result[0] is mock_ep

    @patch('ge.es._plugin_loader.entry_points')
    def test_iter_plugin_entry_points_with_select(self, mock_entry_points):
        """测试 _iter_plugin_entry_points() 带 select 方法的路径"""
        mock_ep = Mock()
        mock_entry_points_obj = Mock()
        mock_entry_points_obj.select.return_value = [mock_ep]
        mock_entry_points.side_effect = [TypeError("test"), mock_entry_points_obj]
        result = list(_iter_plugin_entry_points())
        assert len(result) == 1
        assert result[0] is mock_ep

    @patch('ge.es._plugin_loader.entry_points')
    def test_iter_plugin_entry_points_with_get(self, mock_entry_points):
        """测试 _iter_plugin_entry_points() 使用 get 方法的路径"""
        mock_ep = Mock()
        mock_entry_points_obj = Mock()
        del mock_entry_points_obj.select  
        mock_entry_points_obj.get.return_value = [mock_ep]
        mock_entry_points.side_effect = [TypeError("test"), mock_entry_points_obj]
        result = list(_iter_plugin_entry_points())
        assert len(result) == 1
        assert result[0] is mock_ep

    @patch('ge.es._plugin_loader.entry_points')
    def test_iter_plugin_entry_points_typeerror(self, mock_entry_points):
        """测试 _iter_plugin_entry_points() TypeError 异常路径"""
        mock_entry_points_without_group = Mock()
        mock_entry_points_without_group.select.return_value = [Mock()]
        mock_entry_points.side_effect = [TypeError("test"), mock_entry_points_without_group]
        result = list(_iter_plugin_entry_points())
        assert isinstance(result, list)

    @patch('ge.es._plugin_loader._iter_plugin_entry_points')
    def test_load_all_plugins_without_name(self, mock_iter):
        """测试 load_all_plugins() 处理没有名称的 entry point"""
        mock_ep = Mock()
        type(mock_ep).name = None
        mock_iter.return_value = [mock_ep]
        with patch('ge.es._plugin_loader.debug_print') as mock_debug_print:
            plugins = load_all_plugins()
            assert isinstance(plugins, dict)
            mock_debug_print.assert_called()
            # 检查是否调用了 WARNING 级别的日志
            assert any(call[0][0] == 'WARNING' for call in mock_debug_print.call_args_list)

    @patch('ge.es._plugin_loader._iter_plugin_entry_points')
    def test_load_all_plugins_attribute_error(self, mock_iter):
        """测试 load_all_plugins() 处理 AttributeError"""
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.side_effect = AttributeError("test error")
        mock_iter.return_value = [mock_ep]
        with patch('ge.es._plugin_loader.debug_print') as mock_debug_print:
            plugins = load_all_plugins()
            assert isinstance(plugins, dict)
            mock_debug_print.assert_called()
            # 检查是否调用了 ERROR 级别的日志
            assert any(call[0][0] == 'ERROR' for call in mock_debug_print.call_args_list)

    @patch('ge.es._plugin_loader._iter_plugin_entry_points')
    def test_load_all_plugins_import_error(self, mock_iter):
        """测试 load_all_plugins() 处理 ImportError"""
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.side_effect = ImportError("test import error")
        mock_iter.return_value = [mock_ep]
        with patch('ge.es._plugin_loader.debug_print') as mock_debug_print:
            plugins = load_all_plugins()
            assert isinstance(plugins, dict)
            mock_debug_print.assert_called()
            # 检查是否调用了 ERROR 级别的日志
            assert any(call[0][0] == 'ERROR' for call in mock_debug_print.call_args_list)

    @patch('ge.es._plugin_loader._iter_plugin_entry_points')
    def test_load_all_plugins_general_exception(self, mock_iter):
        """测试 load_all_plugins() 处理一般异常"""
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.side_effect = ValueError("test value error")
        mock_iter.return_value = [mock_ep]
        with patch('ge.es._plugin_loader.debug_print') as mock_debug_print:
            plugins = load_all_plugins()
            assert isinstance(plugins, dict)
            mock_debug_print.assert_called()
            # 检查是否调用了 ERROR 级别的日志
            assert any(call[0][0] == 'ERROR' for call in mock_debug_print.call_args_list)

    @patch('ge.es._plugin_loader._iter_plugin_entry_points')
    def test_load_all_plugins_success(self, mock_iter):
        """测试 load_all_plugins() 成功加载插件"""
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_module = types.ModuleType("test_module")
        mock_module.__name__ = "test_module"
        mock_ep.load.return_value = mock_module
        mock_iter.return_value = [mock_ep]
        with patch('ge.es._plugin_loader.debug_print') as mock_debug_print:
            with patch('sys.modules', {}):
                plugins = load_all_plugins()
                assert isinstance(plugins, dict)
                assert "test_plugin" in plugins
                assert plugins["test_plugin"] is mock_module
                mock_debug_print.assert_called()
                # 检查是否调用了 INFO 级别的日志
                assert any(call[0][0] == 'INFO' for call in mock_debug_print.call_args_list)

