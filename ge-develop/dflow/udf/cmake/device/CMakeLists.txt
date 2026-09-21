# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project (udf_device)

set(CMAKE_CXX_STANDARD 17)

set(PRODUCT_SIDE device)

set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

if (NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH
        ${TOP_DIR}/cmake/modules
    )
endif()

if (NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH
        ${ASCEND_3RD_LIB_PATH}/protoc
        ${ASCEND_3RD_LIB_PATH}/protobuf_static
        ${ASCEND_3RD_LIB_PATH}/ascend_protobuf
        ${ASCEND_3RD_LIB_PATH}/ascend_protobuf_static
        ${ASCEND_INSTALL_PATH}/
        ${ASCEND_INSTALL_PATH}/include
    )
endif()

if (NOT DEFINED DEV_FIND_LIB_PATHS)
    set(DEV_FIND_LIB_PATHS
        ${ASCEND_INSTALL_PATH}/
    )
endif()

if (NOT DEFINED CMAKE_INCLUDE_PATH) 
    set(CMAKE_INCLUDE_PATH  ${ASCEND_INSTALL_PATH}/include ${ASCEND_INSTALL_PATH})
endif()

include(func.cmake)
include(${TOP_DIR}/cmake/intf_pub_linux.cmake)
include(protobuf.cmake)
include(Findaicpu.cmake)
include(Findhccl.cmake)
include(Findmmpa.cmake)
include(Findslog.cmake)
include(Findsecurec.cmake)
include(${TOP_DIR}/cmake/air_headers.cmake)
include(${TOP_DIR}/cmake/modules/Findmetadef.cmake)
include(${TOP_DIR}/cmake/modules/Findascend_hal.cmake)

add_subdirectory(${TOP_DIR}/dflow/udf ${CMAKE_BINARY_DIR}/udf_device)

set(PACKAGE_TAR_FILE_NAME "cann-udf-compat.tar.gz")
set(PACKAGE_BIN_DIR "bin")
set(PACKAGE_LIB_DIR "lib64")

get_target_property(ALOG_PATH alog IMPORTED_LOCATION)
get_filename_component(ALOG_REAL_PATH "${ALOG_PATH}" REALPATH MUST_EXIST)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_BIN_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}

        # 将可执行文件复制到临时目录的 bin 下
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_executor> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_BIN_DIR}/

        # 将库文件复制到临时目录的 lib64 下
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:flow_func> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:reader_writer> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:built_in_flowfunc> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_profiling> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_dump> ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy ${ALOG_REAL_PATH} ${CMAKE_BINARY_DIR}/package_temp/${PACKAGE_LIB_DIR}/

        # 执行 tar 命令进行打包
        # WORKING_DIRECTORY 设置为临时目录的父目录，这样 tar 打包的路径就是相对的
        COMMAND tar -czf ${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME} -C ${CMAKE_BINARY_DIR}/package_temp . --mode=750

        # 打包完成后，删除临时目录
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/package_temp

        # 明确指定这个命令依赖于哪些目标
        DEPENDS udf_executor flow_func reader_writer built_in_flowfunc udf_profiling udf_dump

        COMMENT "Creating udf package: ${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}"
)

add_custom_target(pack_udf_compat ALL
        DEPENDS ${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}
)

if (ENABLE_SIGN)
    sign_file(
            INPUT "${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}"
            CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/sign/sign.xml
            RESULT_VAR "udf_compat_sign_file"
            DEPENDS pack_udf_compat
    )

    install(FILES
            ${udf_compat_sign_file}
            DESTINATION ${CMAKE_INSTALL_PREFIX}
    )

    # pack Ascend-runtime_device-minios.tar.gz
    set(RUNTIME_PACKAGE_TAR_FILE_NAME "Ascend-runtime_device-minios.tar.gz")
    set(RUNTIME_PACKAGE_TAR_PACK_DIR "${CMAKE_BINARY_DIR}/runtime_package_temp")
    add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/${RUNTIME_PACKAGE_TAR_FILE_NAME}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_BIN_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_LIB_DIR}

            # 将可执行文件复制到临时目录的 bin 下
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_executor> ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_BIN_DIR}/

            # 将库文件复制到临时目录的 lib64 下
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:flow_func> ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_LIB_DIR}/
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:reader_writer> ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_LIB_DIR}/
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:built_in_flowfunc> ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_LIB_DIR}/
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_profiling> ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_LIB_DIR}/
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:udf_dump> ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_LIB_DIR}/
            COMMAND ${CMAKE_COMMAND} -E copy ${ALOG_REAL_PATH} ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/${PACKAGE_LIB_DIR}/

            COMMAND ${HI_PYTHON} ${TOP_DIR}/scripts/signtool/image_extract/ci_img_headler.py -img ${ASCEND_INSTALL_PATH}/compat/cann-hccd-compat.tar.gz -raw ${RUNTIME_PACKAGE_TAR_PACK_DIR}/cann-hccd-compat.tar.gz --rcvr
            COMMAND tar -zxf ${RUNTIME_PACKAGE_TAR_PACK_DIR}/cann-hccd-compat.tar.gz -C ${RUNTIME_PACKAGE_TAR_PACK_DIR}/runtime/

            # 执行 tar 命令进行打包
            # WORKING_DIRECTORY 设置为临时目录的父目录，这样 tar 打包的路径就是相对的
            COMMAND tar -czf ${CMAKE_BINARY_DIR}/${RUNTIME_PACKAGE_TAR_FILE_NAME} -C ${RUNTIME_PACKAGE_TAR_PACK_DIR}/ runtime --mode=750

            # 打包完成后，删除临时目录
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${RUNTIME_PACKAGE_TAR_PACK_DIR}

            # 明确指定这个命令依赖于哪些目标
            DEPENDS udf_executor flow_func reader_writer built_in_flowfunc udf_profiling udf_dump

            COMMENT "Creating ${RUNTIME_PACKAGE_TAR_FILE_NAME} package: ${CMAKE_BINARY_DIR}/${RUNTIME_PACKAGE_TAR_FILE_NAME}"
    )
    add_custom_target(pack_runtime_device ALL
            DEPENDS ${CMAKE_BINARY_DIR}/${RUNTIME_PACKAGE_TAR_FILE_NAME}
    )
    sign_file(
            INPUT "${CMAKE_BINARY_DIR}/${RUNTIME_PACKAGE_TAR_FILE_NAME}"
            CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/sign/runtime_dev_sign.xml
            RESULT_VAR "runtime_device_sign_file"
            DEPENDS pack_runtime_device sign_cann_udf_compat
    )

    install(FILES
            ${runtime_device_sign_file}
            DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
else ()
    install(
            FILES "${CMAKE_BINARY_DIR}/${PACKAGE_TAR_FILE_NAME}"
            DESTINATION ${CMAKE_INSTALL_PREFIX}
            COMPONENT application_packaging
    )
endif ()

install(TARGETS flow_func
    LIBRARY DESTINATION device/lib64
)