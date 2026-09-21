# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(ge_api_c_wrapper)
file(GLOB_RECURSE GE_API_C_WRAPPER CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

add_custom_target(ge_python ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ge_py-0.0.1-py3-none-any.whl)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ge_py-0.0.1-py3-none-any.whl
        COMMAND echo "Start to build ge-py wheel package"
        && rm -rf ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/wheel/setup.py
        && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in ${CMAKE_CURRENT_BINARY_DIR}/wheel/
        && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/ge ${CMAKE_CURRENT_BINARY_DIR}/wheel/
        && cp $<TARGET_FILE:graph_wrapper> ${CMAKE_CURRENT_BINARY_DIR}/wheel/ge/_capi/
        && cd ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && ${HI_PYTHON} setup.py bdist_wheel >/dev/null
        && cp -f dist/ge_py-0.0.1-py3-none-any.whl ${CMAKE_CURRENT_BINARY_DIR}/
        && echo "Build ge-py wheel package end"
        DEPENDS eager_style_graph_builder_base ge_api_c_wrapper ${GE_API_C_WRAPPER}
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ge_py-0.0.1-py3-none-any.whl OPTIONAL
        DESTINATION ${INSTALL_LIBRARY_DIR}
)