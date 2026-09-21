backend_e2e_st_test(div_abs_test
    CODEGEN div_abs_backend_generate.cpp
    KERNEL_SRC
        div_abs_test_kernel.cpp
        div_abs_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_div_abs_expect_kernel.cpp)
