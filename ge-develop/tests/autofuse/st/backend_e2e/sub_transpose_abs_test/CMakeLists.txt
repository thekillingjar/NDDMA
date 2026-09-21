backend_e2e_st_test(sub_transpose_abs_test
    CODEGEN sub_transpose_abs_backend_generate.cpp
    KERNEL_SRC
        sub_transpose_abs_test_kernel.cpp
        sub_transpose_abs_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_sub_transpose_abs_expect_kernel.cpp)
