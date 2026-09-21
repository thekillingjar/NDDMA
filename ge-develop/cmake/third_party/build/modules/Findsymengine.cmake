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

set(LIB_FILE "${CMAKE_THIRD_PARTY_LIB_DIR}/symengine/lib") # 编译之后才会有的文件，用于判断是否已经编译
set(MOD_FILE "${CMAKE_THIRD_PARTY_LIB_DIR}/symengine/symengine/mod.cpp") # 打上patch之后才会有的文件，用于判断是否打了patch
set(CMAKE_FILE "${CMAKE_THIRD_PARTY_LIB_DIR}/symengine/CMakeLists.txt") # 用于判断是否已下载并解压
set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/symengine/symengine-0.12.0.tar.gz")
set(SYMENGINE_EXTRA_ARGS "")
if(EXISTS ${LIB_FILE})
    message(STATUS "[symengine] ${LIB_FILE} found, symengine is ready after compile.")
else()
    if(EXISTS ${MOD_FILE})
        message(STATUS "[symengine] ${MOD_FILE} found, symengine is ready with patch installed.")
    elseif(EXISTS ${CMAKE_FILE})
        message(STATUS "[symengine] ${CMAKE_FILE} found, symengine is ready without patch installed.")
        list(APPEND SYMENGINE_EXTRA_ARGS
            PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/symengine_add_mod.patch
        )
    elseif(EXISTS ${REQ_URL})
        message(STATUS "[symengine] ${REQ_URL} found.")
        list(APPEND SYMENGINE_EXTRA_ARGS
            URL ${REQ_URL}
            PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/symengine_add_mod.patch
        )
    else()
        message(STATUS "[symengine] symengine not found, need download.")
        set(REQ_URL "https://gitcode.com/cann-src-third-party/symengine/releases/download/v0.12.0/symengine-0.12.0.tar.gz")
        list(APPEND SYMENGINE_EXTRA_ARGS
            URL ${REQ_URL}
            DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/symengine
            PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/symengine_add_mod.patch
        )
    endif()
    set(SYMENGINE_CXXFLAGS "-fPIC -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI} -std=c++17")

    ExternalProject_Add(symengine_build
            SOURCE_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/symengine
            ${SYMENGINE_EXTRA_ARGS}
            TLS_VERIFY OFF
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
                -DINTEGER_CLASS:STRING=boostmp
                -DBUILD_SHARED_LIBS:BOOL=OFF
                -DBOOST_ROOT=${CMAKE_THIRD_PARTY_LIB_DIR}/boost
                -DBUILD_TESTS=off
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5
                -DCMAKE_CXX_STANDARD=17
                -DWITH_SYMENGINE_THREAD_SAFE:BOOL=ON
                -DCMAKE_CXX_EXTENSIONS=OFF
                -DCMAKE_CXX_FLAGS=${SYMENGINE_CXXFLAGS}
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                -DCMAKE_INSTALL_PREFIX=${CMAKE_THIRD_PARTY_LIB_DIR}/symengine
                -DCMAKE_PREFIX_PATH=${CMAKE_THIRD_PARTY_LIB_DIR}/boost
                -DLLVM_PATH=${LLVM_PATH}
                -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                <SOURCE_DIR>
            BUILD_COMMAND $(MAKE)
            INSTALL_COMMAND $(MAKE) install
            EXCLUDE_FROM_ALL TRUE
            )
    if(NOT BOOST_FOUND)
        add_dependencies(symengine_build boost_build)
    endif()
endif()