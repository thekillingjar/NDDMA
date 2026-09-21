# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(SRC_LIST
    "dnnengines.cc"
    "engine_manager.cc"
)

############ libengine.so ############
add_library(engine SHARED ${SRC_LIST})

add_dependencies(engine
    graphengine_protos
)

target_compile_options(engine PRIVATE
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
    -fvisibility=hidden
)

target_compile_definitions(engine PRIVATE
    REUSE_MEMORY=1
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    FUNC_VISIBILITY
)

target_include_directories(engine PRIVATE
    ${AIR_CODE_DIR}/base
    ${AIR_CODE_DIR}/compiler
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/framework
    ${AIR_CODE_DIR}/inc/external
)

target_link_options(engine PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(engine PRIVATE
    intf_pub
    metadef_headers
    error_manager
    -Wl,--no-as-needed
    c_sec
    unified_dlog
    -Wl,--as-needed
    -lrt
    -ldl
)

install(TARGETS engine OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
