backend_e2e_st_test(floordiv_abs_test
    CODEGEN floordiv_abs_backend_generate.cpp 
    KERNEL_SRC
        floordiv_abs_test_kernel.cpp
        floordiv_abs_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_floordiv_abs_expect_kernel.cpp)
