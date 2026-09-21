backend_e2e_st_test(float_logical_not_test
    CODEGEN float_logical_not_backend_generate.cpp
    KERNEL_SRC
        float_logical_not_test_kernel.cpp
        float_logical_not_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_float_logical_not_expect_kernel.cpp)