# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET gtest_shared_build)
    return()
endif ()

find_package(GTest CONFIG
    PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)

if(GTest_FOUND)
    message(STATUS "[GTest] GTest found: ${GTest_DIR}")
    return()
endif()

message(STATUS "[GTest] GTest not found, finding binary file.")

set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/gtest_shared/googletest-1.14.0.tar.gz")
set(GTEST_EXTRA_ARGS "")
if(EXISTS ${REQ_URL})
    message(STATUS "[GTest] ${REQ_URL} found.")
else()
    message(STATUS "[GTest] ${REQ_URL} not found, need download.")
    set(REQ_URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz")
    list(APPEND GTEST_EXTRA_ARGS
        DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/gtest_shared
    )
endif()

include(ExternalProject)
set(gtest_CXXFLAGS "-fPIC -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
set(GTEST_SHARED_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/gtest_shared)
ExternalProject_Add(gtest_shared_build
    URL ${REQ_URL}
    TLS_VERIFY OFF
    ${GTEST_EXTRA_ARGS}
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DCMAKE_CXX_FLAGS=${gtest_CXXFLAGS}
        -DCMAKE_C_FLAGS=${gtest_CXXFLAGS}
        -DCMAKE_INSTALL_PREFIX=${GTEST_SHARED_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib64
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=ON
        <SOURCE_DIR>
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    EXCLUDE_FROM_ALL TRUE
)
ExternalProject_Add_Step(gtest_shared_build extra_install
    COMMAND ${CMAKE_COMMAND} -E chdir ${GTEST_SHARED_DIR} ${CMAKE_COMMAND} -E create_symlink lib64 ${CMAKE_INSTALL_LIBDIR}
    DEPENDEES install
)
