backend_e2e_st_test(ub_scalar_brc_abs_add_test
    CODEGEN ub_scalar_brc_abs_add_backend_generate.cpp
    KERNEL_SRC
        ub_scalar_brc_abs_add_test_kernel.cpp
        ub_scalar_brc_abs_add_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_ub_scalar_brc_abs_add_expect_kernel.cpp)
