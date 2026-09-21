find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

file(GLOB_RECURSE CODEGEN_TEST_SRCS CONFIGURE_DEPENDS "*.cpp")
add_library(test_py_module OBJECT
        ${CODEGEN_TEST_SRCS}
)
target_link_options(test_py_module PRIVATE
        -Wl,--no-as-needed
)
target_link_libraries(test_py_module PRIVATE
    atrace
    pyautofuse
    graph_base
    graph
    json
    Python3::Python
    GTest::gtest)
