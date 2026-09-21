# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(BENCHMARK_DIR ${CMAKE_THIRD_PARTY_LIB_DIR})
if (TARGET benchmark_build)
    return()
endif()

find_package(benchmark CONFIG
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(benchmark_FOUND)
    message(STATUS "[benchmark] benchmark found: ${benchmark_DIR}")
    return()
endif()

message(STATUS "[benchmark] benchmark not found, finding binary file.")

set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/benchmark/benchmark-1.8.3.tar.gz")
set(BENCHMARK_EXTRA_ARGS "")
if(EXISTS ${REQ_URL})
    message(STATUS "[benchmark] ${REQ_URL} found.")
else()
    message(STATUS "[benchmark] ${REQ_URL} not found, need download.")
    set(REQ_URL "https://gitcode.com/cann-src-third-party/benchmark/releases/download/v1.8.3/benchmark-1.8.3.tar.gz")
    list(APPEND BENCHMARK_EXTRA_ARGS
        DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/benchmark
    )
endif()

include(ExternalProject)
set(benchmark_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI} -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
ExternalProject_Add(benchmark_build
    URL ${REQ_URL}
    TLS_VERIFY OFF
    ${BENCHMARK_EXTRA_ARGS}
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_FLAGS=${benchmark_CXXFLAGS}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_THIRD_PARTY_LIB_DIR}/benchmark
        -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
        -DLLVM_PATH=${LLVM_PATH}
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_MACOSX_RPATH=TRUE
        <SOURCE_DIR>
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    EXCLUDE_FROM_ALL TRUE
)
