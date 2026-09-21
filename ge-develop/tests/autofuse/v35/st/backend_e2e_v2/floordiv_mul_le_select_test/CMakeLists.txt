backend_e2e_st_test(floordiv_mul_le_select_test
    CODEGEN floordiv_mul_le_select_backend_generate.cpp
    KERNEL_SRC
        floordiv_mul_le_select_test_kernel.cpp
        floordiv_mul_le_select_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_floordiv_mul_le_select_expect_kernel.cpp)
