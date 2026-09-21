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

import unittest
from unittest.mock import patch
import os.path
import shutil
import argparse
from dataflow.tools.create_func_ws import main


class TestDataflowCreateFuncWs(unittest.TestCase):
    def test_create_ws(self):
        functions = "Sub:i0:i2:o0,Add:i1:i3:o1:o0"
        ws_dir = "./test_assign_ws"
        clz_name = "Assign"
        with patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(
                    clz_name=clz_name, workspace=ws_dir, functions=functions
                ),
        ):
            with patch("builtins.input", return_value="y"):
                main()
        self.assertTrue(os.path.exists(ws_dir))
        self.assertTrue(os.path.exists(os.path.join(ws_dir, "CMakeLists.txt")))
        self.assertTrue(os.path.exists(os.path.join(ws_dir, "func_assign.json")))
        shutil.rmtree(ws_dir)

    def test_create_ws_confirm_no(self):
        functions = "Sub:i0:i2:o0,Add:i1:i3:o1:o0"
        ws_dir = "./test_assign_ws"
        clz_name = "Assign"
        with patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(
                    clz_name=clz_name, workspace=ws_dir, functions=functions
                ),
        ):
            with patch("builtins.input", return_value="n"):
                main()
        self.assertFalse(os.path.exists(ws_dir))

    def test_create_ws_format_error(self):
        functions = "Sub:i0,Add:i1:o1:o0"
        ws_dir = "./test_assign_ws"
        clz_name = "Assign"
        with patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(
                    clz_name=clz_name, workspace=ws_dir, functions=functions
                ),
        ):
            with patch("builtins.input", return_value="y"):
                self.assertRaises(ValueError, main)

    def test_create_ws_support_without_clz_name(self):
        functions = "Sub:i0:i2:o0,Add:i1:o1:o0"
        ws_dir = "./test_assign_ws"
        with patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(
                    workspace=ws_dir, functions=functions, clz_name=""
                ),
        ):
            with patch("builtins.input", return_value="y"):
                main()
        self.assertTrue(os.path.exists(ws_dir))
        shutil.rmtree(ws_dir)
