backend_e2e_st_test(abs_brc_add_test
    CODEGEN abs_brc_add_backend_generate.cpp
    KERNEL_SRC
        abs_brc_add_test_kernel.cpp
        abs_brc_add_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_abs_brc_add_expect_kernel.cpp)
