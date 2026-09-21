# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET symengine_build)
    return()
endif ()

include(ExternalProject)

set(REQ_URL "https://gitcode.com/cann-src-third-party/symengine/releases/download/v0.12.0/symengine-0.12.0.tar.gz")
set(SYMENGINE_CXXFLAGS  "--std=c++14  -O2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -s -D_FORTIFY_SOURCE=2 -Wno-maybe-uninitialized -Wno-unused-parameter -D_GLIBCXX_USE_CXX11_ABI=0 -Wl,-Bsymbolic")

ExternalProject_Add(symengine_build
        URL ${REQ_URL}
        TLS_VERIFY OFF
        CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -DINTEGER_CLASS:STRING=boostmp
            -DBUILD_SHARED_LIBS:BOOL=ON
            -DBOOST_ROOT=${CMAKE_INSTALL_PREFIX}/boost
            -DBUILD_TESTS=off
            -DCMAKE_CXX_STANDARD=14
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DCMAKE_CXX_EXTENSIONS=OFF
            -DCMAKE_CXX_FLAGS=${SYMENGINE_CXXFLAGS}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/symengine
             <SOURCE_DIR>
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        EXCLUDE_FROM_ALL TRUE
        )
add_dependencies(symengine_build boost_build)