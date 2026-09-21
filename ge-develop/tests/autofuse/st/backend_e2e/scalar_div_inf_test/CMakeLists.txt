backend_e2e_st_test(scalar_div_inf_test
    CODEGEN scalar_div_inf_backend_generate.cpp
    KERNEL_SRC
        scalar_div_inf_test_kernel.cpp
        scalar_div_inf_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_scalar_div_inf_expect_kernel.cpp)
