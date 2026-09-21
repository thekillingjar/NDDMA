# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 设置安装目录
set(PYGE_TEST_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/ge_py_install)

# 安装 ge-py 包到指定目录
# so需要单独拷贝，因为ge_py的安装包中没有打包此so
# 运行环境下因为此so随着Run包安装，放在LD_LIBRARY_PATH中，所以运行环境无问题
add_custom_command(
        OUTPUT ${PYGE_TEST_INSTALL_DIR}/.installed
        COMMAND ${CMAKE_COMMAND} -E remove -rf ${PYGE_TEST_INSTALL_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PYGE_TEST_INSTALL_DIR}
        COMMAND ${HI_PYTHON} -m pip install --target ${PYGE_TEST_INSTALL_DIR} ${CMAKE_BINARY_DIR}/api/python/ge/ge_py-0.0.1-py3-none-any.whl --force-reinstall
        COMMAND ${HI_PYTHON} -m pip install --target ${PYGE_TEST_INSTALL_DIR} ${CMAKE_BINARY_DIR}/tests/ge/ut/ge/graph/eager_style_graph_builder/graph_construction_test/output/whl/es_ut_test-1.0.0-py3-none-any.whl --force-reinstall
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:eager_style_graph_builder_base> ${PYGE_TEST_INSTALL_DIR}/ge/_capi/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/tests/ge/ut/ge/graph/eager_style_graph_builder/graph_construction_test/output/lib64/libes_ut_test.so ${PYGE_TEST_INSTALL_DIR}/ge/_capi/
        COMMAND ${CMAKE_COMMAND} -E touch ${PYGE_TEST_INSTALL_DIR}/.installed
        DEPENDS ge_python es_ut_test
        COMMENT "Installing ge-py and es_ut_test packages for pyge_tests"
)

# 创建安装目标
add_custom_target(install_ge_py_for_tests DEPENDS ${PYGE_TEST_INSTALL_DIR}/.installed)