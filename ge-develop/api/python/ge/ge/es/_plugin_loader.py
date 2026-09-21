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

"""Runtime utilities for dynamically discovering and loading ES plugins."""

import importlib
import sys
from types import ModuleType
from typing import Dict, Any, List

try:
    from importlib.metadata import entry_points
except ImportError:  
    from importlib_metadata import entry_points  # type: ignore

_ENTRY_POINT_GROUP = "ge.es.plugins"

# Log level keywords
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"


def debug_print(level: str, message: str, *args):
    """Print debug message with formatted output.
    
    Args:
        level: Log level (INFO, WARNING, ERROR)
        message: Message format string
        *args: Arguments for message formatting
    """
    module_name = __name__
    formatted_message = message % args if args else message
    print(f"[{level}] [{module_name}] {formatted_message}")


def _iter_plugin_entry_points() -> List[Any]:
    """Compatible with entry_points API changes across different Python versions."""
    try:
        # Python 3.10+
        candidates = entry_points(group=_ENTRY_POINT_GROUP)
        if isinstance(candidates, dict):
            return list(candidates.get(_ENTRY_POINT_GROUP, []))
        return list(candidates)
    except TypeError:
        # Python 3.7-3.9
        candidates = entry_points()
        if hasattr(candidates, "select"):
            return list(candidates.select(group=_ENTRY_POINT_GROUP))
        return list(candidates.get(_ENTRY_POINT_GROUP, []))


def _coerce_to_module(obj: Any, plugin_name: str) -> ModuleType:
    """Convert entry point loading result to a module object.
    
    Args:
        obj: Object returned by entry point loading
        plugin_name: Plugin name for error reporting
        
    Returns:
        ModuleType: Converted module object
        
    Raises:
        TypeError: If the object type is not supported
    """
    if isinstance(obj, ModuleType):
        return obj
    if isinstance(obj, str):
        return importlib.import_module(obj)
    if callable(obj):
        result = obj()
        return _coerce_to_module(result, plugin_name)
    raise TypeError(
        f"Plugin '{plugin_name}' returned unexpected type: {type(obj).__name__}. "
        f"Expected ModuleType, str, or callable returning module."
    )


def load_all_plugins() -> Dict[str, ModuleType]:
    """Load all plugins registered to ge.es.plugins.
    
    Returns:
        dict[str, ModuleType]: Mapping of plugin names to module objects
    """
    plugins: Dict[str, ModuleType] = {}
    
    for entry_point in _iter_plugin_entry_points():
        name = getattr(entry_point, "name", None)
        if not name:
            debug_print(LOG_LEVEL_WARNING, "Ignoring ES plugin entry point without name: %s", entry_point)
            continue
        
        try:
            # Load entry point
            loaded_obj = entry_point.load()
            
            # Convert to module object
            module = _coerce_to_module(loaded_obj, name)
            
            # Register to sys.modules to make import ge.es.<name> available
            fullname = f"ge.es.{name}"
            sys.modules.setdefault(fullname, module)
            
            # Add to plugin dictionary
            plugins[name] = module
            
            debug_print(LOG_LEVEL_INFO, "ES plugin '%s' loaded: %s", name, module.__name__)
            
        except AttributeError as err:
            debug_print(
                LOG_LEVEL_ERROR,
                "Failed to load ES plugin '%s': entry point '%s' missing required attribute. "
                "Ensure the plugin's __init__.py defines get_module(). Error: %s",
                name, getattr(entry_point, "value", "unknown"), err
            )
        except ImportError as err:
            debug_print(
                LOG_LEVEL_ERROR,
                "Failed to import ES plugin '%s': %s. "
                "Check if all dependencies are installed.",
                name, err
            )
        except Exception as err:
            debug_print(
                LOG_LEVEL_ERROR,
                "Unexpected error loading ES plugin '%s' (entry point: %s): %s",
                name, getattr(entry_point, "value", "unknown"), err
            )
    
    return plugins