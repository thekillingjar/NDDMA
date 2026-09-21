# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET grpc_build)
    return()
endif()

include(ExternalProject)

set(GRPC_CONFIG "${CMAKE_THIRD_PARTY_LIB_DIR}/grpc/lib/cmake/grpc/gRPCConfig.cmake") # 编译后的GRPC路径

if(EXISTS ${GRPC_CONFIG})
  message(STATUS "[grpc] ${GRPC_CONFIG} found.")
  set(protobuf_grpc_FOUND true)
else()
  message(STATUS "[grpc] ${GRPC_CONFIG} not found.")
  set(protobuf_grpc_FOUND false)
endif()

if(protobuf_grpc_FOUND)
    message(STATUS "[grpc] protobuf_grpc found, skip compiling.")
else()
    message(STATUS "[grpc] protobuf_grpc not found, finding binary file.")

    set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/grpc/grpc-1.60.0.tar.gz")
    # 初始化可选参数列表
    set(GRPC_EXTRA_ARGS "")
    if(EXISTS ${REQ_URL})
        message(STATUS "[grpc] ${REQ_URL} found.")
    else()
        message(STATUS "[grpc] ${REQ_URL} not found, need download.")
        set(REQ_URL "https://gitcode.com/cann-src-third-party/grpc/releases/download/v1.60.0/grpc-1.60.0.tar.gz")
        list(APPEND GRPC_EXTRA_ARGS
            DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/grpc
        )
    endif()
    
    set(GRPC_CXX_FLAGS "-Wl,-z,relro,-z,now,-z,noexecstack -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -s -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
    ExternalProject_Add(grpc_build
                        URL ${REQ_URL}
                        TLS_VERIFY OFF
                        ${GRPC_EXTRA_ARGS}
                        PATCH_COMMAND ${CMAKE_COMMAND} -E make_directory <SOURCE_DIR>/third_party/opencensus-proto/src
                            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/zlib_bin_build-prefix/src/zlib_bin_build <SOURCE_DIR>/third_party/zlib
                            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/re2_build-prefix/src/re2_build/re2 <SOURCE_DIR>/re2
                            COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/grpc-fix-compile-bug-in-device.patch
                        CONFIGURE_COMMAND ${CMAKE_COMMAND}
                            # zlib
                            -DgRPC_ZLIB_PROVIDER=module
                            # cares
                            -DgRPC_CARES_PROVIDER=module
                            -DCARES_ROOT_DIR=${CMAKE_BINARY_DIR}/cares_build-prefix/src/cares_build
                            -DCARES_BUILD_TOOLS=OFF
                            # re2
                            -DgRPC_RE2_PROVIDER=module
                            -DRE2_ROOT_DIR=${CMAKE_BINARY_DIR}/re2_build-prefix/src/re2_build
                            # absl
                            -DgRPC_ABSL_PROVIDER=module
                            -DABSL_ROOT_DIR=${CMAKE_BINARY_DIR}/abseil_build-prefix/src/abseil_build
                            # protobuf
                            -DgRPC_PROTOBUF_PROVIDER=module
                            -DPROTOBUF_ROOT_DIR=${CMAKE_BINARY_DIR}/protobuf_static_build-prefix/src/protobuf_static_build
                            -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
                            # ssl
                            -DgRPC_SSL_PROVIDER=package
                            -DOPENSSL_ROOT_DIR=${CMAKE_THIRD_PARTY_LIB_DIR}/openssl
                            -DOPENSSL_USE_STATIC_LIBS=TRUE
                            # grpc option
                            -DCMAKE_CXX_STANDARD=14
                            -DCMAKE_CXX_FLAGS=${GRPC_CXX_FLAGS}
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DCMAKE_BUILD_TYPE=Release
                            -DgRPC_BUILD_TESTS=OFF
                            -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                            -DLLVM_PATH=${LLVM_PATH}
                            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                            -DgRPC_BUILD_CSHARP_EXT=OFF
                            -DgRPC_BUILD_CODEGEN=OFF
                            -DgRPC_BUILD_GRPC_CPP_PLUGIN=OFF
                            -DCMAKE_INSTALL_PREFIX=${CMAKE_THIRD_PARTY_LIB_DIR}/grpc
                            <SOURCE_DIR>
                        BUILD_COMMAND $(MAKE)
                        INSTALL_COMMAND $(MAKE) install && ${CMAKE_COMMAND} -E touch ${CMAKE_THIRD_PARTY_LIB_DIR}/grpc/lib/cmake/grpc/gRPCPluginTargets.cmake
                        EXCLUDE_FROM_ALL TRUE
    )

    add_dependencies(grpc_build openssl_build re2_build zlib_bin_build cares_build abseil_build protobuf_static_build)
endif()