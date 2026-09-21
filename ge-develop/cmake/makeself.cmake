# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# makeself.cmake - 自定义 makeself 打包脚本

# 设置 makeself 路径
set(MAKESELF_EXE ${CPACK_MAKESELF_PATH}/makeself.sh)
set(MAKESELF_HEADER_EXE ${CPACK_MAKESELF_PATH}/makeself-header.sh)
if(NOT MAKESELF_EXE)
    message(FATAL_ERROR "makeself not found! Install it with: sudo apt install makeself")
endif()

# 创建临时安装目录
set(STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging")
file(MAKE_DIRECTORY "${STAGING_DIR}")

# 执行安装到临时目录
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}"
    RESULT_VARIABLE INSTALL_RESULT
)

message("Remove files from ${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging/lib")
file(GLOB ALL_FILES "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging/lib/*")  # * 匹配所有文件/目录

# 删除lib目录下所有install的文件，这些是各个cmake中install的，子包中不应该在这个目录下面，参见package.cmake中新install的内容
file(REMOVE_RECURSE 
    "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging/lib"  # 递归删除目录及内容
)
    
if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation to staging directory failed: ${INSTALL_RESULT}")
endif()
# 生成安装配置文件
set(CSV_OUTPUT ${CPACK_CMAKE_BINARY_DIR}/filelist.csv)

message(STATUS "CPACK_BUILD_COMPONENT: ${CPACK_BUILD_COMPONENT}")
execute_process(
    COMMAND python3 ${CPACK_CMAKE_SOURCE_DIR}/scripts/package/package.py --pkg_name ${CPACK_BUILD_COMPONENT} --os_arch linux-${CPACK_ARCH}
    WORKING_DIRECTORY ${CPACK_CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE result
    ERROR_VARIABLE error
    RESULT_VARIABLE code
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "package.py result: ${code}")
if (NOT code EQUAL 0)
    message(FATAL_ERROR "Filelist generation failed: ${result}")
else ()
    message(STATUS "Filelist generated successfully: ${result}")

    if (NOT EXISTS ${CSV_OUTPUT})
        message(FATAL_ERROR "Output file not created: ${CSV_OUTPUT}")
    endif ()
endif ()
set(SCENE_OUT_PUT
    ${CPACK_CMAKE_BINARY_DIR}/scene.info
)
configure_file(
    ${SCENE_OUT_PUT}
    ${STAGING_DIR}/share/info/${CPACK_BUILD_COMPONENT}/
    COPYONLY
)
set(OPS_VERSION_OUT_PUT
    ${CPACK_CMAKE_BINARY_DIR}/${CPACK_BUILD_COMPONENT}_version.h
)
configure_file(
    ${OPS_VERSION_OUT_PUT}
    ${STAGING_DIR}/share/info/${CPACK_BUILD_COMPONENT}/
    COPYONLY
)
configure_file(
    ${CSV_OUTPUT}
    ${STAGING_DIR}/share/info/${CPACK_BUILD_COMPONENT}/script
    COPYONLY
)

# makeself打包
file(STRINGS ${CPACK_CMAKE_BINARY_DIR}/makeself.txt script_output)
string(REPLACE " " ";" makeself_param_string "${script_output}")

list(LENGTH makeself_param_string LIST_LENGTH)
math(EXPR INSERT_INDEX "${LIST_LENGTH} - 2")
list(INSERT makeself_param_string ${INSERT_INDEX} "${STAGING_DIR}")

message(STATUS "script output: ${script_output}")
message(STATUS "makeself: ${makeself_param_string}")

execute_process(COMMAND bash ${MAKESELF_EXE}
        --header ${MAKESELF_HEADER_EXE}
        --help-header share/info/${CPACK_BUILD_COMPONENT}/script/help.info
        ${makeself_param_string} share/info/${CPACK_BUILD_COMPONENT}/script/install.sh
        WORKING_DIRECTORY ${STAGING_DIR}
        RESULT_VARIABLE EXEC_RESULT
        ERROR_VARIABLE  EXEC_ERROR
)

if(NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself packaging failed: ${EXEC_ERROR}")
endif()
