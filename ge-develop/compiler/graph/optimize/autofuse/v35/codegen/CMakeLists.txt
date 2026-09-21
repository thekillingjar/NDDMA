file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp")

target_sources(aihac_codegen PRIVATE
        ${SOURCES}
)

target_include_directories(aihac_codegen PRIVATE
        ${CODE_ROOT_DIR}/v35/
        ${CODE_ROOT_DIR}/v35/codegen
)

target_link_libraries(aihac_codegen PUBLIC
        $<BUILD_INTERFACE:ascendc_api_regbase_extend>
        $<BUILD_INTERFACE:ascendc_api_cube_extend>
)