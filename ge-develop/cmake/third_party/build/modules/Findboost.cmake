# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET boost_build)
    return()
endif ()

set(Boost_ROOT "${CMAKE_THIRD_PARTY_LIB_DIR}/boost/")
set(Boost_CONFIG_PATH "${CMAKE_THIRD_PARTY_LIB_DIR}/boost/lib/cmake/Boost-1.87.0/")

find_package(Boost CONFIG
      PATHS ${CMAKE_THIRD_PARTY_LIB_DIR}
      NO_DEFAULT_PATH)

if(Boost_FOUND)
    message(STATUS "[boost] boost found: ${Boost_INCLUDE_DIRS}")
    ExternalProject_Add(boost_build
                        SOURCE_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/boost
                        CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "")
    return()
endif()

message(STATUS "[boost] boost not found, finding binary file.")
include(ExternalProject)
set(BOOST_CONFIG_BINARY "${CMAKE_THIRD_PARTY_LIB_DIR}/boost/tools/boost_install/BoostConfig.cmake") # 未编译的boost路径
set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/boost/boost_1_87_0.tar.gz")
set(BOOST_EXTRA_ARGS "")
if(EXISTS ${BOOST_CONFIG_BINARY})
  message(STATUS "[boost] ${BOOST_CONFIG_BINARY} found.")
  list(APPEND BOOST_EXTRA_ARGS
      SOURCE_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/boost
  )
elseif(EXISTS ${REQ_URL})
  message(STATUS "[boost] ${REQ_URL} found.")
  list(APPEND BOOST_EXTRA_ARGS
      URL ${REQ_URL}
  )
else()
  message(STATUS "[boost] ${REQ_URL} not found, need download.")
  set(REQ_URL "https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz")
  list(APPEND BOOST_EXTRA_ARGS
      URL ${REQ_URL}
      DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/boost
  )
endif()
ExternalProject_Add(boost_build
                    TLS_VERIFY OFF
                    ${BOOST_EXTRA_ARGS}
                    CONFIGURE_COMMAND  cd <SOURCE_DIR> && sh bootstrap.sh --prefix=${CMAKE_THIRD_PARTY_LIB_DIR}/boost --with-libraries=headers
                    BUILD_COMMAND   cd <SOURCE_DIR> &&  ./b2 headers install
                    INSTALL_COMMAND ""
                    EXCLUDE_FROM_ALL TRUE
)