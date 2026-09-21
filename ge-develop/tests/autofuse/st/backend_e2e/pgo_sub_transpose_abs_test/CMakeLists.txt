backend_e2e_st_test(pgo_sub_transpose_abs_test
    CODEGEN pgo_sub_transpose_abs_backend_generate.cpp
    KERNEL_SRC
        pgo_sub_transpose_abs_test_kernel.cpp
        pgo_sub_transpose_abs_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_pgo_sub_transpose_abs_expect_kernel.cpp)
