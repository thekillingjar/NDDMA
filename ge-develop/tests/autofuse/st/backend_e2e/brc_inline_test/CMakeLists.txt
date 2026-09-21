backend_e2e_st_test(brc_inline_test
    CODEGEN brc_inline_backend_generate.cpp 
    KERNEL_SRC
        brc_inline_test_kernel.cpp
        brc_inline_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_brc_inline_expect_kernel.cpp)
