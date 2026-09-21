# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
###for profiling so, This so cannot be deleted for compatibility purposes
set(udf_profiling_files
    udf_profiling_manager.cpp)
##for so inc path
set(profiling_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/external
)

add_library(udf_profiling SHARED
    ${udf_profiling_files}
)
target_include_directories(udf_profiling PRIVATE
    ${profiling_inc_path}
)

target_compile_options(udf_profiling PRIVATE
    -ftrapv
    -O2
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_link_libraries(udf_profiling PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<IF:$<STREQUAL:${PRODUCT_SIDE},device>,alog,unified_dlog>
)

set(INSTALL_LIBRARY_DIR lib)

install(TARGETS udf_profiling OPTIONAL
    EXPORT udf_profiling-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}
)
