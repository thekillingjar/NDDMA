file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp")

target_sources(aihac_codegen PRIVATE
        ${SOURCES}
)

target_include_directories(aihac_codegen PRIVATE
        ${CODE_ROOT_DIR}/v35/optimize
)
