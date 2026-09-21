# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    set(Python3_EXECUTABLE ${HI_PYTHON})
    find_package(Python3 COMPONENTS Interpreter Development)
    if (Python3_FOUND)
        set(HI_PYTHON_INC ${Python3_INCLUDE_DIRS})
        cmake_print_variables(HI_PYTHON_INC)
    else ()
        execute_process(COMMAND ${HI_PYTHON} -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE HI_PYTHON_INC
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (result)
            message(WARNING "no include dir found for python:${HI_PYTHON}.")
        endif ()
    endif ()
    # make sure pybind11 cmake can be use in find_package
    execute_process(COMMAND ${HI_PYTHON} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED)
else ()
    set(pybind11_INCLUDE_DIR ${HI_PYTHON_SITE_PACKAGE}/pybind11/include)
endif ()
cmake_print_variables(HI_PYTHON_INC)
cmake_print_variables(pybind11_INCLUDE_DIR)