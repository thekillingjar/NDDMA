# ===== 前置条件 =====
# 已经安装RUN包
# 设置环境变量

# ===== 基础设置 =====
cmake_minimum_required(VERSION 3.16)
project(sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(
        _GLIBCXX_USE_CXX11_ABI=0
        google=ascend_private
)

set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
set(ASCEND_LIB_DIR ${ASCEND_INSTALL_PATH}/lib64)
message(STATUS "ASCEND_LIB_DIR = ${ASCEND_LIB_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_PREFIX_PATH  ${ASCEND_LIB_DIR} ${CMAKE_PREFIX_PATH})

# ===== 引入函数（推荐：使用 find_package） =====
list(APPEND CMAKE_MODULE_PATH "${ASCEND_INSTALL_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)


# ===== 1. 前置条件：定义原型库 =====
# 注意: 
# 当前为开箱即用示例, 即定义INTERFACE使用已有的原型so
# 正常使用不会是用现有的so，更普遍的方式是直接使用项目中已有的so的target
# 使用项目中已有的so方式:
# 1. add_library(opgraph_all SHARED)
# 2. 设置PROPERTIES LIBRARY_OUTPUT_DIRECTORY
add_library(opgraph_all INTERFACE)
set_target_properties(opgraph_all PROPERTIES
    INTERFACE_LIBRARY_OUTPUT_DIRECTORY "${ASCEND_INSTALL_PATH}/opp/built-in/op_proto"
)

# ===== 2. 生成 ES API 包 =====
# 排除 Conv2D, ConcatD生成
add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_all
    OPP_PROTO_TARGET  opgraph_all
    OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
    EXCLUDE_OPS       Conv2D,ConcatD
)

# ===== 2.1. 自定义 ES API =====
file(GLOB CPP_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/custom/*.cpp")
add_library(self_defined_es_lib SHARED ${CPP_FILES})

find_library(GRAPH_LIB NAMES graph)

# 添加头文件路径
target_include_directories(self_defined_es_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/custom
)

# 将API包link到es_all中
target_link_libraries(self_defined_es_lib PUBLIC es_all)

add_dependencies(self_defined_es_lib es_all)
# ===== 3. 在应用中使用 =====
add_executable(sample src/main.cc)
target_link_libraries(sample PRIVATE
    ${GRAPH_LIB}
    self_defined_es_lib # 自动获得依赖、头文件和库
    c_sec
)
