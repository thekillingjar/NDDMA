backend_e2e_st_test(add_abs_float_scalar_test
    CODEGEN add_abs_float_scalar_backend_generate.cpp 
    KERNEL_SRC
        add_abs_float_scalar_test_kernel.cpp
        add_abs_float_scalar_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_add_abs_float_scalar_expect_kernel.cpp)
