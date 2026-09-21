backend_e2e_st_test(cast_isfinite_test
    CODEGEN cast_isfinite_backend_generate.cpp
    KERNEL_SRC
        cast_isfinite_test_kernel.cpp
        cast_isfinite_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_cast_isfinite_expect_kernel.cpp)
