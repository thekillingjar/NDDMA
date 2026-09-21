backend_e2e_st_test(brc_reduce_test
        CODEGEN brc_reduce_backend_generate.cpp
        KERNEL_SRC
        brc_reduce_test_kernel.cpp
        brc_reduce_test_tiling.cpp
        autofuse_tiling_data.h
        TEST_SRC test_e2e_brc_reduce_expect_kernel.cpp)
