# FindGenerateEsPackage.cmake
# CMake MODULE 模式查找脚本
#
# 用于 find_package(GenerateEsPackage) 查找 ES API 生成模块
#
# 使用方式:
#   find_package(GenerateEsPackage REQUIRED)
#   add_es_library_and_whl(...)
#
# 前提：
#   - 查找的camke在同一路径

# 获取当前查找脚本所在的目录
get_filename_component(_FIND_GENERATE_ES_PACKAGE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)

# 包含实际的函数定义文件
include("${_FIND_GENERATE_ES_PACKAGE_DIR}/generate_es_package.cmake")

# 标记模块已找到
set(GenerateEsPackage_FOUND TRUE)

# 设置版本信息
set(GenerateEsPackage_VERSION "1.0.0")

# 设置模块提供的函数列表（用于文档和调试）
set(GenerateEsPackage_FUNCTIONS "add_es_library_and_whl;add_es_library")

# 输出查找成功的信息
if (NOT GenerateEsPackage_FIND_QUIETLY)
    message(STATUS "Found GenerateEsPackage module (version ${GenerateEsPackage_VERSION})")
    message(STATUS "  - Module directory: ${_FIND_GENERATE_ES_PACKAGE_DIR}")
    message(STATUS "  - Provides functions: ${GenerateEsPackage_FUNCTIONS}")
endif ()

# 处理 REQUIRED 和 QUIET 参数
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GenerateEsPackage
    REQUIRED_VARS GenerateEsPackage_FOUND
    VERSION_VAR GenerateEsPackage_VERSION
)

