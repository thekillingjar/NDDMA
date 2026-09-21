cmake_minimum_required(VERSION 3.16)

project(nddma_data_collection)

set(ASC_SRC_FILE nddma_data_collection.asc)
set(NDDMA_STAGE "2" CACHE STRING "NDDMA sensitivity binary id: 1 dtype/dim, 2 bytes/shape, 3 input stride, 4 output stride, 5 alignment, 6 broadcast")
set(NPU_ARCH "dav-3510" CACHE STRING "Ascend NPU architecture passed to ASC compiler")
if(NOT DEFINED CMAKE_ASC_STANDARD OR "${CMAKE_ASC_STANDARD}" STREQUAL "")
    set(CMAKE_ASC_STANDARD 17 CACHE STRING "ASC language standard" FORCE)
endif()
set(CMAKE_ASC_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../.." ABSOLUTE)
set(REGBASE_INCLUDE_DIR
    "${REPO_ROOT}/ge-develop/compiler/graph/optimize/autofuse/v35/ascendc/api_regbase"
    CACHE PATH "Directory containing datacopy_nddma.h"
)

if("${RUN_MODE}" STREQUAL "cpu")
    find_package(tikicpulib REQUIRED)
    set_source_files_properties(${ASC_SRC_FILE} PROPERTIES LANGUAGE CXX)
else()
    find_package(ASC REQUIRED)
    enable_language(ASC CXX)
endif()

add_executable(demo_nddma
    ${ASC_SRC_FILE}
)

target_include_directories(demo_nddma PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${REGBASE_INCLUDE_DIR}
)

target_compile_definitions(demo_nddma PRIVATE
    NDDMA_STAGE=${NDDMA_STAGE}
)

target_compile_options(demo_nddma PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:-g -O0>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-x c++>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall>>
)

target_link_directories(demo_nddma PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:$ENV{ASCEND_HOME_PATH}/x86_64-linux/simulator/${SOC_VERSION}/lib>>
)

target_link_libraries(demo_nddma PRIVATE
    tiling_api
    register
    platform
    m
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:npu_drv>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:cpudebug_acl_stub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:tikicpulib::ascend950pr_9599>>
)
