# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cc" "*.CC" "*.cpp" "*.CPP")

add_executable(ge_graph_dsl_test ${SOURCES})

target_include_directories(ge_graph_dsl_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${AIR_CODE_DIR}/inc
)

target_compile_options(ge_graph_dsl_test PRIVATE
    -g
)
set_target_properties(ge_graph_dsl_test PROPERTIES CXX_STANDARD 17)

target_link_libraries(ge_graph_dsl_test PUBLIC GTest::gtest ge_graph_dsl_b)

include(CTest)
enable_testing()
add_test(NAME test COMMAND ge_graph_dsl_test)