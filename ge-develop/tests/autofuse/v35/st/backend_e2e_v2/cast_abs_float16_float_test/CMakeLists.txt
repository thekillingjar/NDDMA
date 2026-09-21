backend_e2e_st_test(cast_abs_float16_float_test
    CODEGEN cast_abs_float16_float_backend_generate.cpp 
    KERNEL_SRC
        cast_abs_float16_float_test_kernel.cpp
        cast_abs_float16_float_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_cast_abs_float16_float_expect_kernel.cpp)
