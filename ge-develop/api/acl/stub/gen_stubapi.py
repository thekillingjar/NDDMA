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

import os
import re
import sys

PATTERN_FUNCTION = re.compile(r'ACL_FUNC_VISIBILITY\s+\n+.+\w+\([^();]*\);|.+\w+\([^();]*\);')
PATTERN_RETURN = re.compile(r'([^ ]+[ *])\w+\([^;]+;')

RETURN_STATEMENTS = {
    'aclDataType':          '    return ACL_DT_UNDEFINED;',
    'aclFormat':            '    return ACL_FORMAT_UNDEFINED;',
    'aclError':             '    printf("[ERROR]: stub library cannot be used for execution, please check your \
environment variables and compilation options to make sure you use the correct ACL library.\\n");\n    \
return static_cast<aclError>(ACL_ERROR_COMPILING_STUB_MODE);',
    'void':                 '',
    'size_t':               '    return static_cast<size_t>(0);',
    'uint8_t':              '    return static_cast<uint8_t>(0);',
    'int32_t':              '    return static_cast<int32_t>(0);',
    'uint32_t':             '    return static_cast<uint32_t>(0);',
    'int64_t':              '    return static_cast<int64_t>(0);',
    'uint64_t':             '    return static_cast<uint64_t>(0);',
    'aclFloat16':           '    return static_cast<aclFloat16>(0);',
    'float':                '    return 0.0f;',
    'bool':                 '    return false;',
    'double':               '    return static_cast<double>(0.0f);',
}


def collect_header_files(cblas_inc_dir, op_compiler_inc_dir, op_exec_inc_dir, mdl_inc_dir):
    """input path,return relevant header files"""
    cblas_headers = []
    op_compiler_headers = []
    op_exec_headers = []
    mdl_headers = []
    for root, dirs, files in os.walk(cblas_inc_dir):
        files.sort()
        for file in files:
            if file.find("cblas") >= 0:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')
                cblas_headers.append(file_path)
    for root, dirs, files in os.walk(op_compiler_inc_dir):
        files.sort()
        for file in files:
            if file.find("op_compiler") >= 0:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')
                op_compiler_headers.append(file_path)
    for root, dirs, files in os.walk(op_exec_inc_dir):
        files.sort()
        for file in files:
            if file.find("acl_op.h") >= 0:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')
                op_exec_headers.append(file_path)
    for root, dirs, files in os.walk(mdl_inc_dir):
        files.sort()
        for file in files:
            if file.find("mdl.h") >= 0:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')
                mdl_headers.append(file_path)
    return cblas_headers, op_compiler_headers, op_exec_headers, mdl_headers


def collect_functions(file_path):
    signatures = []
    with open(file_path) as f:
        content = f.read()
        matches = PATTERN_FUNCTION.findall(content)
        for signature in matches:
            signatures.append(signature)
    return signatures


def implement_function(func):
    # remove ';'
    function_def = func[:len(func) - 1]
    function_def += '\n'
    function_def += '{\n'
    m = PATTERN_RETURN.search(func)
    if m:
        ret_type = m.group(1).strip()
        if RETURN_STATEMENTS.__contains__(ret_type):
            function_def += RETURN_STATEMENTS[ret_type]
        else:
            if ret_type.endswith('*'):
                function_def += '    return nullptr;'
            else:
                raise RuntimeError("[ERROR] Unhandled return type: " + ret_type)
    else:
        function_def += '    return nullptr;'
    function_def += '\n'
    function_def += '}\n'
    return function_def


def generate_stub_file(cblas_inc_dir, op_compiler_inc_dir, op_exec_inc_dir, mdl_inc_dir):
    """input inc_dir and return relevant contents"""
    cblas_header_files, op_compiler_header_files, op_exec_header_files, mdl_header_files = collect_header_files(
        cblas_inc_dir, op_compiler_inc_dir, op_exec_inc_dir, mdl_inc_dir)
    print("header files has been generated")
    cblas_content = generate_function(cblas_header_files, cblas_inc_dir)
    print("cblas_content has been generate")
    op_compiler_content = generate_function(op_compiler_header_files, op_compiler_inc_dir)
    print("op_compiler_content has been generate")
    op_exec_content = generate_function(op_exec_header_files, op_exec_inc_dir)
    print("op_exec_content has been generate")
    mdl_content = generate_function(mdl_header_files, mdl_inc_dir)
    print("mdl_content has been generate")
    return cblas_content, op_compiler_content, op_exec_content, mdl_content


def generate_function(header_files, inc_dir):
    includes = []
    includes.append('#include <stdio.h>\n')
    includes.append('#include <stdint.h>\n')
    # generate includes
    for header in header_files:
        if not header.endswith('.h'):
            continue
        include_str = '#include "{}"\n'.format(header[len(inc_dir):])
        includes.append(include_str)

    content = includes
    print("include concent build success")
    total = 0
    content.append('\n')
    # generate implement
    for header in header_files:
        if not header.endswith('.h'):
            continue
        content.append("// stub for {}\n".format(header[len(inc_dir):]))
        functions = collect_functions(header)
        print("inc file:{}, functions numbers:{}".format(header, len(functions)))
        total += len(functions)
        for func in functions:
            content.append("{}\n".format(implement_function(func)))
            content.append("\n")
    print("implement concent build success")
    print('total functions number is {}'.format(total))
    return content


def gen_code(cblas_inc_dir, op_compiler_inc_dir, op_exec_inc_dir, mdl_inc_dir,
             cblas_stub_path, op_compiler_stub_path, op_exec_stub_path, mdl_stub_path):
    """input inc_dir and relevant cpp files"""
    if not cblas_inc_dir.endswith('/'):
        cblas_inc_dir += '/'
    if not op_compiler_inc_dir.endswith('/'):
        op_compiler_inc_dir += '/'
    if not op_exec_inc_dir.endswith('/'):
        op_exec_inc_dir += '/'
    if not mdl_inc_dir.endswith('/'):
        mdl_inc_dir += '/'
    cblas_content, op_compiler_content, op_exec_content, mdl_content = generate_stub_file(
        cblas_inc_dir, op_compiler_inc_dir, op_exec_inc_dir, mdl_inc_dir)
    print("cblas_content, op_compiler_content, op_exec_content, mdl_content have been generated")
    with open(cblas_stub_path, mode='w') as f:
        f.writelines(cblas_content)
    with open(op_compiler_stub_path, mode='w') as f:
        f.writelines(op_compiler_content)
    with open(op_exec_stub_path, mode='w') as f:
        f.writelines(op_exec_content)
    with open(mdl_stub_path, mode='w') as f:
        f.writelines(mdl_content)

if __name__ == '__main__':
    cblas_inc_dir = sys.argv[1]
    op_compiler_inc_dir = sys.argv[2]
    op_exec_inc_dir = sys.argv[3]
    mdl_inc_dir = sys.argv[4]
    cblas_stub_file = sys.argv[5]
    op_compiler_stub_file = sys.argv[6]
    op_exec_stub_file = sys.argv[7]
    mdl_stub_file = sys.argv[8]
    gen_code(cblas_inc_dir, op_compiler_inc_dir, op_exec_inc_dir, mdl_inc_dir,
             cblas_stub_file, op_compiler_stub_file, op_exec_stub_file, mdl_stub_file)
