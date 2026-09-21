backend_e2e_st_test(scalar_brc_test
    CODEGEN scalar_brc_backend_generate.cpp
    KERNEL_SRC
        scalar_brc_test_kernel.cpp
        scalar_brc_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_scalar_brc_expect_kernel.cpp)
