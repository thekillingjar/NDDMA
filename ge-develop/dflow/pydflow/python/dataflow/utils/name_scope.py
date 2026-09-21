#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

__all__ = ["name_scope", "reset_name_scope"]

import threading


class _NameScope(object):
    def __init__(self, name):
        self.name = name
        self.namespace = None
        self.auto_name = True
        self._old_name_scope = []
        self._names_in_used = {}

    def __enter__(self):
        self._old_name_scope.append(self.name)
        return self._unique_name()

    def __exit__(self, type_arg, value_arg, traceback_arg):
        self._old_name_scope.pop()

    def _unique_name(self):
        name_key = "/".join(self._old_name_scope)
        names = self._names_in_used.get(self.namespace, None)
        i = 0
        if names is not None:
            i = names.get(name_key, 0)

        if names is not None and i > 0:
            if not self.auto_name:
                raise ValueError(f"name:{name_key} is already used, please modify.")
            name = "%s_%d" % (name_key, i)
            current_scope = "%s_%d" % (self._old_name_scope.pop(), i)
            self._old_name_scope.append(current_scope)
        else:
            name = name_key
        if self.namespace not in self._names_in_used:
            self._names_in_used[self.namespace] = {}
        self._names_in_used[self.namespace][name_key] = i + 1
        return name


_thread_local_data = threading.local()
_thread_local_data.name_scope = _NameScope(None)


def name_scope(name, default_name=None, namespace=None, auto_name=True):
    name = default_name if name is None else name
    if not isinstance(name, str):
        raise TypeError("name for name_scope must be a string.")
    if namespace is not None and not isinstance(namespace, str):
        raise TypeError("namespace for name_scope must be a string.")
    global _thread_local_data
    _thread_local_data.name_scope.name = name
    _thread_local_data.name_scope.namespace = namespace
    _thread_local_data.name_scope.auto_name = auto_name
    return _thread_local_data.name_scope


def reset_name_scope():
    global _thread_local_data
    _thread_local_data = threading.local()
    _thread_local_data.name_scope = _NameScope(None)
