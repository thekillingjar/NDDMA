backend_e2e_st_test(cast_abs_test
    CODEGEN cast_abs_backend_generate.cpp 
    KERNEL_SRC
        cast_abs_test_kernel.cpp
        cast_abs_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_cast_abs_expect_kernel.cpp)
