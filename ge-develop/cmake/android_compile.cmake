# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CPU_TYPE aarch64)
set(TARGET_LINUX_DISTRIBUTOR_ID lhisilinux)
set(TARGET_LINUX_DISTRIBUTOR_RELEASE 200)

set(CMAKE_C_COMPILER "$ENV{ANDROID_TOOLCHAIN_PATH}/gcc/linux-x86/aarch64/aarch64-himix200-linux/bin/aarch64-mix210-linux-gcc" CACHE PATH "C Compiler") 
set(CMAKE_CXX_COMPILER "$ENV{ANDROID_TOOLCHAIN_PATH}/gcc/linux-x86/aarch64/aarch64-himix200-linux/bin/aarch64-mix210-linux-g++" CACHE PATH "C++ Compiler")
set(CMAKE_LINKER "$ENV{ANDROID_TOOLCHAIN_PATH}/gcc/linux-x86/aarch64/aarch64-himix200-linux/bin/aarch64-mix210-linux-g++" CACHE PATH "LINKER") 
set(CMAKE_AR "$ENV{ANDROID_TOOLCHAIN_PATH}/gcc/linux-x86/aarch64/aarch64-himix200-linux/bin/aarch64-mix210-linux-ar" CACHE PATH "AR")
set(CMAKE_RANLIB "$ENV{ANDROID_TOOLCHAIN_PATH}/gcc/linux-x86/aarch64/aarch64-himix200-linux/bin/aarch64-mix210-linux-ranlib" CACHE PATH "RANLIB") 
set(CMAKE_STRIP "$ENV{ANDROID_TOOLCHAIN_PATH}/gcc/linux-x86/aarch64/aarch64-himix200-linux/bin/aarch64-mix210-linux-strip" CACHE PATH "STRIP")

set(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> <DEFINES> -D__FILE__='\"$(notdir $(abspath <SOURCE>))\"' -Wno-builtin-macro-redefined  <INCLUDES> <FLAGS> -o <OBJECT>  -c <SOURCE>")
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES>  -D__FILE__='\"$(notdir $(abspath <SOURCE>))\"' -Wno-builtin-macro-redefined  <INCLUDES> <FLAGS> -o <OBJECT>  -c <SOURCE>")

#remove default options for cmake
set (CMAKE_C_FLAGS_DEBUG "" CACHE STRING "c debug flag" FORCE)
set (CMAKE_C_FLAGS_RELEASE "" CACHE STRING "c release flag" FORCE)
set (CMAKE_C_FLAGS_RELWITHDEBINFO "" CACHE STRING "c relwith flag" FORCE)
set (CMAKE_C_FLAGS_MINSIZEREL "" CACHE STRING "c minsize flag" FORCE)

set (CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "cxx debug flag" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "cxx release flag" FORCE)
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE STRING "cxx relwith flag" FORCE)
set (CMAKE_CXX_FLAGS_MINSIZEREL "" CACHE STRING "cxx minsize flag" FORCE)

set (CMAKE_ASM_FLAGS_DEBUG "" CACHE STRING "asm debug flag" FORCE)
set (CMAKE_ASM_FLAGS_RELEASE "" CACHE STRING "asm release flag" FORCE)
set (CMAKE_ASM_FLAGS_RELWITHDEBINFO "" CACHE STRING "asm relwith flag" FORCE)
set (CMAKE_ASM_FLAGS_MINSIZEREL "" CACHE STRING "asm minsize flag" FORCE)

#删除静态库中的时间戳
set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> qcD <TARGET> <LINK_FLAGS> <OBJECTS>")
set(CMAKE_C_ARCHIVE_APPEND "<CMAKE_AR> qD <TARGET> <LINK_FLAGS> <OBJECTS>")
set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET> ")

set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcD <TARGET> <LINK_FLAGS> <OBJECTS>")
set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> qD <TARGET> <LINK_FLAGS> <OBJECTS>")
set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET> ")


set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)
