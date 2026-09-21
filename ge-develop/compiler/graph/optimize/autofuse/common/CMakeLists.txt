file(GLOB_RECURSE CODEGEN_COMMON_SRCS CONFIGURE_DEPENDS "*.cpp" "*.cc")

target_include_directories(aihac_codegen PRIVATE
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        ${TOP_DIR}/runtime/include/external
)

target_sources(aihac_codegen PRIVATE
    ${CODEGEN_COMMON_SRCS}
)
