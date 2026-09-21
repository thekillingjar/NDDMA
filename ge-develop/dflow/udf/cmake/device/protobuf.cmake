# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(ExternalProject)
include(GNUInstallDirs)

set(PROTOBUF_SRC_DIR ${CMAKE_BINARY_DIR}/protobuf-src)
set(PROTOBUF_DL_DIR ${CMAKE_BINARY_DIR}/downloads)
set(PROTOBUF_STATIC_PKG_DIR ${CMAKE_BINARY_DIR}/protobuf_static)
set(PROTOBUF_SHARED_PKG_DIR ${CMAKE_BINARY_DIR}/protobuf_shared)
set(PROTOBUF_HOST_STATIC_PKG_DIR ${CMAKE_BINARY_DIR}/protobuf_host_static)

set(SECURITY_COMPILE_OPT "-Wl,-z,relro,-z,now,-z,noexecstack -s -Wl,-Bsymbolic")
set(PROTOBUF_CXXFLAGS "${SECURITY_COMPILE_OPT} -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -Dgoogle=ascend_private")
set(HOST_PROTOBUF_CXXFLAGS "${SECURITY_COMPILE_OPT} -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -Dgoogle=ascend_private")

if (PRODUCT_SIDE STREQUAL "device")
  message(STATUS "[UDF protobuf] Building for device side")
  set(HOST_PROTOBUF_CXXFLAGS ${PROTOBUF_CXXFLAGS})
endif()

set(HOST_PROTOBUF_CXXFLAGS "${HOST_PROTOBUF_CXXFLAGS}")
message(STATUS "[UDF protobuf] HOST_PROTOBUF_CXXFLAGS is ${HOST_PROTOBUF_CXXFLAGS}")

# 使用设备端工具链生成 ascend_protobuf_static
set(CMAKE_CXX_COMPILER_ ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
set(CMAKE_C_COMPILER_ ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
set(SOURCE_DIR ${PROTOBUF_SRC_DIR})

set(PROTOBUF_PATH ${OPEN_SOURCE_DIR}/protobuf)
set(ABSEIL_PATH ${OPEN_SOURCE_DIR}/abseil-cpp)
set(PATCH_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if ((NOT EXISTS "${PROTOBUF_PATH}/protobuf-all-25.1.tar.gz" AND NOT EXISTS "${PROTOBUF_PATH}/protobuf-25.1.tar.gz") OR
      NOT EXISTS "${ABSEIL_PATH}/abseil-cpp-20230802.1.tar.gz")
  message(STATUS "[UDF protobuf] protobuf tar.gz not exists or abseil-cpp-20230802.1.tar.gz not exists")
  set(REQ_URL "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz")
  set(ABS_REQ_URL "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz")

  ExternalProject_Add(protobuf_src_dl
    URL               ${REQ_URL}
    TLS_VERIFY        OFF
    DOWNLOAD_DIR      ${PROTOBUF_DL_DIR}/
    DOWNLOAD_NO_EXTRACT 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )

  ExternalProject_Add(abseil_src_dl
    URL               ${ABS_REQ_URL}
    TLS_VERIFY        OFF
    DOWNLOAD_DIR      ${PROTOBUF_DL_DIR}/abseil-cpp/
    DOWNLOAD_NO_EXTRACT 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )

  # 下载/解压 protobuf 源码
  ExternalProject_Add(protobuf_src
    DOWNLOAD_COMMAND ""
    COMMAND tar -zxf ${PROTOBUF_DL_DIR}/protobuf-25.1.tar.gz --strip-components 1 -C ${SOURCE_DIR}
    COMMAND tar -zxf ${PROTOBUF_DL_DIR}/abseil-cpp/abseil-cpp-20230802.1.tar.gz --strip-components 1 -C ${SOURCE_DIR}/third_party/abseil-cpp
    PATCH_COMMAND cd ${SOURCE_DIR} && patch -p1 < ${PATCH_PATH}/protobuf_25.1_change_version.patch && cd ${SOURCE_DIR}/third_party/abseil-cpp && patch -p1 < ${PATCH_PATH}/protobuf-hide_absl_symbols.patch
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )


  add_dependencies(protobuf_src protobuf_src_dl abseil_src_dl)
else()
  if(EXISTS "${PROTOBUF_PATH}/protobuf-all-25.1.tar.gz")
    set(PROTOBUF_FILE_NAME "protobuf-all-25.1.tar.gz")
  else()
    set(PROTOBUF_FILE_NAME "protobuf-25.1.tar.gz")
  endif()
  message(STATUS "[UDF protobuf] protobuf tar.gz exists: ${PROTOBUF_FILE_NAME}")
  ExternalProject_Add(protobuf_src
      DOWNLOAD_COMMAND ""
      COMMAND tar -zxf ${OPEN_SOURCE_DIR}/protobuf/${PROTOBUF_FILE_NAME} --strip-components 1 -C ${SOURCE_DIR}
      COMMAND tar -zxf ${OPEN_SOURCE_DIR}/abseil-cpp/abseil-cpp-20230802.1.tar.gz --strip-components 1 -C ${SOURCE_DIR}/third_party/abseil-cpp
      PATCH_COMMAND cd ${SOURCE_DIR} && patch -p1 < ${PATCH_PATH}/protobuf_25.1_change_version.patch && cd ${SOURCE_DIR}/third_party/abseil-cpp && patch -p1 < ${PATCH_PATH}/protobuf-hide_absl_symbols.patch
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
  )
endif()

ExternalProject_Add(protobuf_static_build
  DEPENDS protobuf_src
  SOURCE_DIR ${PROTOBUF_SRC_DIR}
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
      -G ${CMAKE_GENERATOR}
      -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DTOOLCHAIN_DIR=${TOOLCHAIN_DIR}
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DBUILD_SHARED_LIBS=OFF
      -Dprotobuf_WITH_ZLIB=OFF
      -DLIB_PREFIX=ascend_
      -DCMAKE_SKIP_RPATH=TRUE
      -Dprotobuf_BUILD_TESTS=OFF
      -DCMAKE_CXX_STANDARD=14
      -DCMAKE_CXX_FLAGS=${HOST_PROTOBUF_CXXFLAGS}
      -DCMAKE_INSTALL_PREFIX=${PROTOBUF_STATIC_PKG_DIR}
      -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
      -DABSL_COMPILE_OBJ=TRUE
      <SOURCE_DIR>
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
  EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Add(protobuf_shared_build
  DEPENDS protobuf_src
  SOURCE_DIR ${PROTOBUF_SRC_DIR}
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
      -G ${CMAKE_GENERATOR}
      -DTOOLCHAIN_DIR=${TOOLCHAIN_DIR}
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
      -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DBUILD_SHARED_LIBS=ON
      -Dprotobuf_WITH_ZLIB=OFF
      -DLIB_PREFIX=ascend_
      -DCMAKE_SKIP_RPATH=TRUE
      -Dprotobuf_BUILD_TESTS=OFF
      -DCMAKE_CXX_STANDARD=14
      -DCMAKE_CXX_FLAGS=${HOST_PROTOBUF_CXXFLAGS}
      -DCMAKE_INSTALL_PREFIX=${PROTOBUF_SHARED_PKG_DIR}
      -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
      <SOURCE_DIR>
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
  EXCLUDE_FROM_ALL TRUE
)

ExternalProject_Add(protobuf_host_static_build
  DEPENDS protobuf_src
  SOURCE_DIR ${PROTOBUF_SRC_DIR}
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
      -G ${CMAKE_GENERATOR}
      -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DBUILD_SHARED_LIBS=OFF
      -Dprotobuf_WITH_ZLIB=OFF
      -DLIB_PREFIX=host_ascend_
      -DCMAKE_SKIP_RPATH=TRUE
      -Dprotobuf_BUILD_TESTS=OFF
      -DCMAKE_CXX_STANDARD=14
      -DCMAKE_CXX_FLAGS=${HOST_PROTOBUF_CXXFLAGS}
      -DCMAKE_INSTALL_PREFIX=${PROTOBUF_HOST_STATIC_PKG_DIR}
      -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
      -DABSL_COMPILE_OBJ=TRUE
      <SOURCE_DIR>
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
  EXCLUDE_FROM_ALL TRUE
)

set(PROTOBUF_HOST_DIR ${CMAKE_BINARY_DIR}/protobuf_host)
ExternalProject_Add(protobuf_host_build
    DEPENDS protobuf_src
    SOURCE_DIR ${PROTOBUF_SRC_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_INSTALL_PREFIX=${PROTOBUF_HOST_DIR}
        -Dprotobuf_BUILD_TESTS=OFF
        -DCMAKE_CXX_STANDARD=14
        -Dprotobuf_WITH_ZLIB=OFF
        <SOURCE_DIR>
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    EXCLUDE_FROM_ALL TRUE
)

add_executable(host_protoc IMPORTED)
set_target_properties(host_protoc PROPERTIES
    IMPORTED_LOCATION ${PROTOBUF_HOST_DIR}/bin/protoc
)
add_dependencies(host_protoc protobuf_host_build)

add_library(ascend_protobuf_static_lib STATIC IMPORTED)
set_target_properties(ascend_protobuf_static_lib PROPERTIES
    IMPORTED_LOCATION ${PROTOBUF_STATIC_PKG_DIR}/lib/libascend_protobuf.a
)

add_library(ascend_protobuf_static INTERFACE)
target_include_directories(ascend_protobuf_static INTERFACE ${PROTOBUF_STATIC_PKG_DIR}/include)
target_link_libraries(ascend_protobuf_static INTERFACE ascend_protobuf_static_lib)
add_dependencies(ascend_protobuf_static protobuf_static_build)
