backend_e2e_st_test(scalar_float_inf_test
    CODEGEN scalar_float_inf_backend_generate.cpp 
    KERNEL_SRC
        scalar_float_inf_test_kernel.cpp
        scalar_float_inf_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_scalar_float_inf_expect_kernel.cpp)
