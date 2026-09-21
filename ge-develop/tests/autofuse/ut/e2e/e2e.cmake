function(do_add_e2e_test)
    set(one_value_arg
        WORKDIR # Workdir
        )
    set(mul_value_arg
        TILING # Tiling codegen library source file
        CODEGEN # Codegen library source file
        KERNEL_SRC # Kernel source file that codegen will generate
        TEST_SRC # Test case source file
        )

    set(TEST_NAME ${ARGV0})
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${one_value_arg}" "${mul_value_arg}")

    foreach(file ${ARG_KERNEL_SRC})
        list(APPEND KERNEL_SRC "${ARG_WORKDIR}/${file}")
    endforeach()

    add_library(${TEST_NAME}_tiling_gen SHARED ${ARG_TILING})
    target_include_directories(${TEST_NAME}_tiling_gen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
    target_link_libraries(${TEST_NAME}_tiling_gen codegen)

    add_executable(${TEST_NAME}_codegen ${ARG_CODEGEN})
    target_link_libraries(${TEST_NAME}_codegen ${TEST_NAME}_tiling_gen codegen e2e aihac_symbolizer metadef) #aihac_symbolizer

    add_custom_command(OUTPUT ${KERNEL_SRC}
                       WORKING_DIRECTORY ${ARG_WORKDIR}
                       COMMAND ${TEST_NAME}_codegen
                       DEPENDS ${TEST_NAME}_codegen)

    add_executable(${TEST_NAME} ${KERNEL_SRC} ${ARG_TEST_SRC})
    target_include_directories(${TEST_NAME} PRIVATE ${ARG_WORKDIR})
    target_link_libraries(${TEST_NAME} tikicpulib_ascend910B1 metadef GTest::gtest GTest::gtest_main)

    gtest_discover_tests(${TEST_NAME})
endfunction()

macro(add_e2e_test)
    do_add_e2e_test(${ARGV}
        WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
endmacro()
