backend_e2e_st_test(int16_logical_not_test
    CODEGEN int16_logical_not_backend_generate.cpp
    KERNEL_SRC
        int16_logical_not_test_kernel.cpp
        int16_logical_not_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_int16_logical_not_expect_kernel.cpp)