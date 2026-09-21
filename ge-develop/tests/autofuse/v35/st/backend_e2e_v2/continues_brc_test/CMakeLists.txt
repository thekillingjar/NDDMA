backend_e2e_st_test(continues_brc_test
    CODEGEN continues_brc_backend_generate.cpp
    KERNEL_SRC
        continues_brc_test_kernel.cpp
        continues_brc_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_continues_brc_expect_kernel.cpp)
