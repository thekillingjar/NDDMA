cmake_minimum_required(VERSION 3.14)
project(recomand_network_inference)

if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/cann)
endif ()

set(common_compile_options
        -std=c++17
        -Wall
        -D_GLIBCXX_USE_CXX11_ABI=0  # 使用旧的 ABI，确保与依赖库兼容
)

message(STATUS "Ascend Toolkit path: ${ASCEND_PATH}")

set(ASCEND_INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/src
        ${ASCEND_PATH}/include
        ${ASCEND_PATH}/include/ge
        ${ASCEND_PATH}/include/parser
        ${ASCEND_PATH}/include/acl
)

set(ASCEND_LIB_DIRS
        ${ASCEND_PATH}/lib64
)

set(ASCEND_LIBS
        ge_compiler
        graph
        graph_base
        ge_common
        ge_common_base
        ascendcl
        ge_runner
        fmk_parser
)

# 检查库是否存在
foreach (LIB ${ASCEND_LIBS})
    find_library(${LIB}_PATH ${LIB} PATHS ${ASCEND_LIB_DIRS})
    if (NOT ${LIB}_PATH)
        message(FATAL_ERROR "Required library not found: ${LIB}")
    else ()
        message(STATUS "Found library: ${LIB} at ${${LIB}_PATH}")
    endif ()
endforeach ()

function(setup_target target_name)
    target_include_directories(${target_name} PRIVATE ${ASCEND_INCLUDE_DIRS})
    target_link_directories(${target_name} PRIVATE ${ASCEND_LIB_DIRS})
    target_link_libraries(${target_name} PRIVATE ${ASCEND_LIBS})
    target_compile_options(${target_name} PRIVATE ${common_compile_options})
endfunction()

add_library(model_inference SHARED
        src/model_inference.cpp
)

setup_target(model_inference)

# 生成可执行文件，并链接上面的共享库
add_executable(recomand_exec
        src/recomand_random_input.cpp   # 这里放你的入口文件
)
setup_target(recomand_exec)
target_link_libraries(recomand_exec PRIVATE model_inference)