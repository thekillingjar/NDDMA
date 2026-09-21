backend_e2e_st_test(sub_abs_test
    CODEGEN sub_abs_backend_generate.cpp
    KERNEL_SRC
        sub_abs_test_kernel.cpp
        sub_abs_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_sub_abs_expect_kernel.cpp)
