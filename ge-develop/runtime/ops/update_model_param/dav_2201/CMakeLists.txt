project(UpdateModelParam_dav_2201)

# 为update_model_param.h设置语言为CXX
set_source_files_properties(
    update_model_param.h PROPERTIES LANGUAGE CXX
)

if(BUILD_PKG_COMPONENT)
    set(CCEC_PATH ${ASCEND_INSTALL_PATH}/tools/ccec_compiler/bin)
    set(TILING_DEPENDS "")
    set(INCLUDE_PATHS
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/include/
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/impl
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/impl/basic_api
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/impl/simt_api
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/impl/micro_api
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/include/basic_api
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/include/simit_api
        -I${ASCEND_INSTALL_PATH}/x86_64-linux/asc/include/micro_api
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/include/
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/impl
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/impl/basic_api
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/impl/simt_api
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/impl/micro_api
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/include/basic_api
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/include/simit_api
        -I${ASCEND_INSTALL_PATH}/aarch64-linux/asc/include/micro_api
        -I${ASCEND_INSTALL_PATH}/include/
    )
else()
    set(tiling_script ${TOP_DIR}/asc/asc-devkit/impl/adv_api/cmake/scripts/gen_kernel_tiling_data_def.py)
    set(TILING_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(tiling_gen_file ${TILING_GEN_DIR}/kernel_tiling/kernel_tiling.h)
    set(tiling_data_def_path ${TOP_DIR}/asc/asc-devkit/include/adv_api)
    set(TILING_DEPENDS ${tiling_gen_file})

    add_custom_command(OUTPUT ${tiling_gen_file}
        COMMAND python3 ${tiling_script} ${tiling_data_def_path} ${tiling_gen_file}
        DEPENDS ${tiling_script} ${tiling_data_def_path})
    set(INCLUDE_PATHS
        -I${TOP_DIR}/asc/asc-devkit/
        -I${TOP_DIR}/asc/asc-devkit/include/
        -I${TOP_DIR}/asc/asc-devkit/impl
        -I${TOP_DIR}/asc/asc-devkit/impl/basic_api
        -I${TOP_DIR}/asc/asc-devkit/impl/simt_api
        -I${TOP_DIR}/asc/asc-devkit/impl/micro_api
        -I${TOP_DIR}/asc/asc-devkit/include/basic_api
        -I${TOP_DIR}/asc/asc-devkit/include/simit_api
        -I${TOP_DIR}/asc/asc-devkit/include/micro_api
        -I${TILING_GEN_DIR}
    )
endif()

# 添加自定义目标
add_custom_target(runtime_update_model_param_dav_2201
    COMMAND ${CCEC_PATH}/ccec -c -O3 -x cce -DTILING_KEY_VAR=0UL -Dupdate_model_param=UpdateModelParam
        --cce-disable-kernel-global-attr-check --cce-auto-sync -mllvm -api-deps-filter --cce-aicore-arch=dav-c220-vec
        --cce-aicore-only -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false -std=c++17
        ${INCLUDE_PATHS}
        ${CMAKE_CURRENT_SOURCE_DIR}/update_model_param.h
        -o ${CMAKE_CURRENT_BINARY_DIR}/UpdateModelParam_0_dav_2201.o
    COMMAND ${CCEC_PATH}/ld.lld -m aicorelinux -r -Ttext=0
        -x ${CMAKE_CURRENT_BINARY_DIR}/UpdateModelParam_0_dav_2201.o -static
        -o ${CMAKE_CURRENT_BINARY_DIR}/UpdateModelParam_dav_2201.o
    COMMAND ${CCEC_PATH}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/UpdateModelParam_dav_2201.o
        -static -o ${CMAKE_CURRENT_BINARY_DIR}/UpdateModelParam_dav_2201.o
    DEPENDS ${TILING_DEPENDS}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/UpdateModelParam_dav_2201.o
    DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)