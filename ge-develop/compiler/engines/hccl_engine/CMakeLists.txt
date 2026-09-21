# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16.0)
option(BUILD_OPEN_PROJECT "Build open hcce project." ON)
message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")

project(hcce)

add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_COMPILE_OBJECT
	    "<CMAKE_CXX_COMPILER> <DEFINES> -D__FILE__='\"$(notdir $(abspath <SOURCE>))\"' -Wno-builtin-macro-redefined <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>"
	)
	set(CMAKE_C_COMPILE_OBJECT
	    "<CMAKE_C_COMPILER> <DEFINES> -D__FILE__='\"$(notdir $(abspath <SOURCE>))\"' -Wno-builtin-macro-redefined <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>"
 	)

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)

    add_subdirectory(hcom_graph_adaptor)
    add_subdirectory(hcom_gradient_split_tune)

endif()