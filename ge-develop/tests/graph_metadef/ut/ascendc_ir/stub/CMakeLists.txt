include(${GE_METADEF_DIR}/graph/ascendc_ir/generator/generator.cmake)
add_library(ascir_stub_builtin_ops SHARED ascir_stub_builtin_ops.cc ascir_stub_builtin_ops_v2.cc)
target_compile_options(ascir_stub_builtin_ops PRIVATE
        -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_options(ascir_stub_builtin_ops PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        -L ${INSTALL_BASE_DIR}/lib
        )
target_link_libraries(ascir_stub_builtin_ops PRIVATE aihac_ir_register)
target_include_directories(ascir_stub_builtin_ops PRIVATE
        ${METADEF_DIR}
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/external
        ${GE_METADEF_DIR}/graph/ascendc_ir/generator
        )
message(${CMAKE_BINARY_DIR})
set(ops_header_dir ${CMAKE_BINARY_DIR}/ascir_stub_builtin_ops)
set_target_properties(ascir_ops_header_generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ops_header_dir})
message("ops_header_dir is " ${ops_header_dir}/ascir_ops.h)
ascir_generate(ascir_stub_builtin_ops
        ${ops_header_dir}
        ${CMAKE_BINARY_DIR}/tests/graph_metadef/ut/ascendc_ir/stub/libascir_stub_builtin_ops.so
        ${ops_header_dir}/ascir_ops.h)

add_custom_target(ascir_stub_builtin_ops_header ALL DEPENDS ${ops_header_dir}/ascir_ops.h)
add_library(ascir_stub_ops_headers INTERFACE)
target_include_directories(ascir_stub_ops_headers INTERFACE ${ops_header_dir})
add_dependencies(ascir_stub_ops_headers ascir_stub_builtin_ops_header)
