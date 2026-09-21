backend_e2e_st_test(cast_nan_test
    CODEGEN cast_nan_backend_generate.cpp
    KERNEL_SRC
        cast_nan_test_kernel.cpp
        cast_nan_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_cast_nan_expect_kernel.cpp)
