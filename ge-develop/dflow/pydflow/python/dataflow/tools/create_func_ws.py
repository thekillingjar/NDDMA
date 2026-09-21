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

import argparse
import os.path

import dataflow.utils.log as log
from dataflow.tools.func_ws_creator import FuncWsCreator


def main():
    parser = argparse.ArgumentParser(description="flow func workspace Creator")
    parser.add_argument(
        "-f",
        "--functions",
        required=True,
        type=str,
        help="Input function info, input and output indexes. e.g. Sub:i0:i2:o0,Add:i1:i3:o1:o0",
    )
    parser.add_argument(
        "-c",
        "--clz_name",
        required=False,
        default="",
        type=str,
        help="flow func class name",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        required=False,
        default="",
        type=str,
        help="flow func workspace path",
    )
    args = parser.parse_args()

    print(
        f"args: functions={args.functions}, clz_name={args.clz_name}, workspace={args.workspace}"
    )
    path = os.path.abspath(args.workspace)
    confirmation = input(
        f"will create workspace in path '{path}', please enter Yes(y) to confirm:"
    )
    if confirmation.lower() == "yes" or confirmation.lower() == "y":
        print("create function workspace begin")
        creator = FuncWsCreator(args.functions, args.clz_name, args.workspace)
        creator.generate()
        print("create function workspace finish")
    else:
        print("exit")


if __name__ == "__main__":
    main()
