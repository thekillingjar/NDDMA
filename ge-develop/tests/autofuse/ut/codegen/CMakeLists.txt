file(GLOB_RECURSE CODEGEN_TEST_SRCS CONFIGURE_DEPENDS "*.cpp")
add_library(test_codegen OBJECT
        ${CODEGEN_TEST_SRCS}
)
target_compile_definitions(test_codegen PRIVATE
        CMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\"
        ASCEND_INSTALL_PATH=\"${ASCEND_INSTALL_PATH}\"
)
target_include_directories(test_codegen PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/../../../../tests/autofuse/depends/runtime/src
)
target_link_libraries(test_codegen PRIVATE
    -Wl,--no-as-needed
    optimize
    codegen
    ascir_builtin_ops
    ascir_ops_headers
    aihac_ir
    graph_base
    graph
    ascir
    ascgen_common
    aihac_symbolizer
    json
    ascendc_api_extend
    GTest::gtest)
