backend_e2e_st_test(int32_logical_not_test
    CODEGEN int32_logical_not_backend_generate.cpp
    KERNEL_SRC
        int32_logical_not_test_kernel.cpp
        int32_logical_not_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_int32_logical_not_expect_kernel.cpp)