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

import sys
import os
import glob
import argparse
import re
from autofuse.compile_adapter import jit_compile
PYF_PATH = os.path.dirname(os.path.realpath(__file__))


def test_mode_0(graph_name, output_file, output_path):
    from input_ascir import tiling_def, host_impl, device_impl

    # # fix kernel code
    # with open("/home/zgj/ascir_tool/kernel_code/tmp_ascir_reduce_sum_arar_4dim_align_2025-06-10_12_48_33/build/device/fused_graph_0_arar_op_kernel.cpp",'r') as kernel_file:
    #     device_impl=kernel_file.read()

    # # fix tiling code
    # with open("/home/zgj/kernel_meta_3964229480019244343/te_ascbackend_028bd3240b9a690b2abb872945c94976a931cecbe8156038522f11aff6859a7adtmt8jh1/host/fused_graph_tiling_func.cpp",'r') as tiling_file:
    #     host_impl=tiling_file.read()

    # # fix def code
    # with open("/home/zgj/kernel_meta_3964229480019244343/te_ascbackend_028bd3240b9a690b2abb872945c94976a931cecbe8156038522f11aff6859a7adtmt8jh1/host/fused_graph_tiling_data.h",'r') as def_file:
    #     tiling_def=def_file.read()

    #off_sync = "--compile_options=--cce-auto-sync=off"
    jit_compile(tiling_def, host_impl, device_impl, [graph_name, output_file, output_path, "--force_unknown=True"])


def process_static_shape_kernel_proc(kernel_file):
    # 定义正则表达式模式
    pattern = re.compile(r'^extern\s+"C"\s+__global__\s+__aicore__\s+void\s+(\w+)\s*\(([^)]*)\)\s*{')
    with open(kernel_file, 'r') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        # 检查是否需要删除包含特定字符串的行
        if 'const AutofuseTilingData t;' in line.strip():
            continue

        match = pattern.match(line)
        if match:
            func_name = match.group(1)
            params_str = match.group(2).strip()
            if not params_str:
                # 无参数的情况，直接添加原行
                result.append(line)
                continue

            params = [p.strip() for p in params_str.split(',')]
            if params and params[-1] == 'AutofuseTilingData param':
                # 修改最后一个参数
                params[-1] = 'AutofuseTilingData t'
                new_params = ', '.join(params)
                new_line = f'extern "C" __global__ __aicore__ void {func_name}({new_params}) '
                new_line += '{\n'
                result.append(new_line)
            else:
                # 不处理，直接添加原行
                result.append(line)
        else:
            result.append(line)

        # 写入处理后的内容到新文件
        with open(kernel_file, 'w') as f:
            f.writelines(result)


def process_kernel_file(kernel_file):
    tilingdata_init_file = PYF_PATH + "/init_tilingdata.h"
    with open(tilingdata_init_file, 'r') as header_file:
        header_content = header_file.read()
    process_static_shape_kernel_proc(kernel_file)
    with open(kernel_file,'r') as file:
        content = file.read()
        if 'GET_TILING_DATA' in content:
            device_impl = header_content + content
            # todo 解析kernel函数，动态生成AutofuseLaunch相关接口
        else:
            device_impl = content
    return device_impl


def process_tiling_file(tiling_file):
    with open(tiling_file,'r') as f:
        content = f.read()
        pattern = r'extern\s+"C"\s+int64_t\s+AutofuseLaunch'
        if re.search(pattern, content):
            return content

    with open(tiling_file,'r') as file:
        lines = file.readlines()

    # 遍历每一行，寻找匹配的行
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 替换函数参数并插入结构体定义
        if re.search(r'extern\s+"C"\s+int64_t\s+AutofuseTiling\(', line):
            new_line = line.replace("uint32_t aiv_num, uint32_t ub_size", "ResLimit *res_limit = nullptr")
            print("old declar : ", lines[i])
            print("new declar : ", new_line)
            lines[i] = new_line

            struct_definition = 'struct ResLimit {  uint32_t valid_num = 0;  uint32_t aiv_num = 0;  int32_t aic_num = 0;  uint32_t resv[11];};\n'
            struct_declaration = 'constexpr ResLimit g_no_limit_res = {1, 48, 0, {}};\n'
            lines.insert(i, struct_definition)
            lines.insert(i + 1, struct_declaration)
            i += 3  # 跳过新插入的两行
            continue
        
        # 替换 block_dim 的设置
        if re.search(r'tiling->set_block_dim\((aiv_num)\)', line):
            new_line = line.replace("tiling->set_block_dim(aiv_num)", "tiling->set_block_dim(limit->aiv_num)")
            print("old block_dim : ", lines[i])
            print("new block_dim : ", new_line)
            lines[i] = new_line
            
            res_limit_check = '  const ResLimit *limit = (res_limit == nullptr || res_limit->aiv_num == 0) ? &g_no_limit_res : res_limit;\n'
            lines.insert(i, res_limit_check)
            i += 2  # 跳过新插入的一行
            continue
        
        # 替换 ub_size 的设置
        if re.search(r'tiling->set_ub_size\((ub_size)\)', line):
            new_line = line.replace("tiling->set_ub_size(ub_size)", "tiling->set_ub_size(1024*192 - 256)")
            print("old ub_size : ", lines[i])
            print("new ub_size : ", new_line)
            lines[i] = new_line
            i += 1
            continue

        # 替换 AutofuseTiling调用 的设置
        if re.search(r'auto\s+ret\s+=\s+AutofuseTiling\(', line):
            new_line = line.replace("aiv_num, (uint32_t)ub_size", "nullptr")
            print("old call : ", lines[i])
            print("new call : ", new_line)
            lines[i] = new_line
            i += 1
            continue
        
        # 默认情况
        i += 1

    host_impl = ''.join(lines)
    return host_impl


def test_mode_1(args, graph_name, output_file, output_path):
    code_path = args.code_path
    host_code_path = args.code_path + "/host/"
    device_code_path = args.code_path + "/device/"
    tiling_def_file = args.code_path + "/host/autofuse_tiling_data.h"
    with open(tiling_def_file,'r') as def_file:
        tiling_def = def_file.read()

    # 构造模糊搜索模式，查找所有以op_kernel.cpp结尾的文件
    device_pattern = os.path.join(device_code_path, "*_op_kernel.cpp")
    # 使用glob.glob查找所有匹配的文件
    device_matching_files = glob.glob(device_pattern, recursive=True)
    if len(device_matching_files) != 1:
        print("Error: No or multiple matching files found for device_impl")
        sys.exit(1)
    print("kernel file :", device_matching_files[0])
    device_impl = process_kernel_file(device_matching_files[0])

    # 构造模糊搜索模式，查找所有以op_kernel.cpp结尾的文件
    host_pattern = os.path.join(host_code_path, "*_tiling_func.cpp")
    # 使用glob.glob查找所有匹配的文件
    host_matching_files = glob.glob(host_pattern, recursive=True)
    if len(host_matching_files) != 1:
        print("Error: No or multiple matching files found for device_impl")
        sys.exit(1)
    print("tiling file :", host_matching_files[0])
    host_impl = process_tiling_file(host_matching_files[0])

    print("################# start compile.")
    jit_compile(tiling_def, host_impl, device_impl, [graph_name, output_file, output_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=int, required=True, help='mode')
    parser.add_argument('--graph_name', default='autofuse', type=str, help='Graph name.')
    parser.add_argument('--code_path', type=str, help='generate code path')
    parser.add_argument('--output_path', type=str, help='save code and compile tmp path')

    # 解析命令行参数
    args = parser.parse_args()

    graph_name = "--graph_name=" + args.graph_name
    output_file = "--output_file=" + args.output_path + "/" + args.graph_name + ".so"
    tmp_path = os.path.join(PYF_PATH, "..", "tmp_ascir")
    output_path = "--output_path=" + tmp_path
    print("graph_name:", graph_name)
    print("output_file:", output_file)
    if args.mode == 0:
        test_mode_0(graph_name, output_file, output_path)
    elif args.mode == 1:
        test_mode_1(args, graph_name, output_file, output_path)
