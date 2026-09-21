# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET dataflow_python)
    message(STATUS "Target dataflow_python already exists.")
    return()
endif ()

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake/modules)
    find_package(udf MODULE REQUIRED)
endif ()

add_custom_target(dataflow_python ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dataflow-0.0.1-py3-none-any.whl)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dataflow-0.0.1-py3-none-any.whl
        COMMAND echo "package dataflow whl start"
        && mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/python/. ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && cp -r ${CMAKE_CURRENT_BINARY_DIR}/wrapper/dflow_wrapper.so ${CMAKE_CURRENT_BINARY_DIR}/wheel/dataflow
        && cp -r ${CMAKE_CURRENT_BINARY_DIR}/wrapper/flow_func_wrapper/flowfunc_wrapper.so ${CMAKE_CURRENT_BINARY_DIR}/wheel/dataflow/flow_func
        && cp -r ${CMAKE_CURRENT_BINARY_DIR}/wrapper/data_wrapper.so ${CMAKE_CURRENT_BINARY_DIR}/wheel/dataflow
        && cd ${CMAKE_CURRENT_BINARY_DIR}/wheel
        && ${HI_PYTHON} setup.py bdist_wheel >/dev/null
        && cp -f dist/dataflow-0.0.1-py3-none-any.whl ${CMAKE_CURRENT_BINARY_DIR}/
        && echo "package dataflow whl end"
        DEPENDS dflow_wrapper flowfunc_wrapper data_wrapper
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dataflow-0.0.1-py3-none-any.whl OPTIONAL DESTINATION ${INSTALL_LIBRARY_DIR})

add_subdirectory(wrapper)
