backend_e2e_st_test(uint8_logical_not_test
    CODEGEN uint8_logical_not_backend_generate.cpp
    KERNEL_SRC
        uint8_logical_not_test_kernel.cpp
        uint8_logical_not_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_uint8_logical_not_expect_kernel.cpp)