# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# add_library(ge_graph_dsl_inc INTERFACE)
# target_include_directories(ge_graph_dsl_inc INTERFACE ./)
# target_link_libraries(ge_graph_dsl_inc INTERFACE easy_graph)

# 定义接口库 ge_graph_dsl_inc
add_library(ge_graph_dsl_inc_b INTERFACE)

# 添加 ge_graph_dsl_inc 的包含目录
target_include_directories(ge_graph_dsl_inc_b INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# 声明对 easy_graph 的依赖
target_link_libraries(ge_graph_dsl_inc_b INTERFACE easy_graph_b)