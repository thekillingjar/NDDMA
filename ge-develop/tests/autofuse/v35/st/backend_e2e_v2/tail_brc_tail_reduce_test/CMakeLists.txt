backend_e2e_st_test(tail_brc_tail_reduce_test
        CODEGEN tail_brc_tail_reduce_backend_generate.cpp
        KERNEL_SRC
        tail_brc_tail_reduce_test_kernel.cpp
        tail_brc_tail_reduce_test_tiling.cpp
        autofuse_tiling_data.h
        TEST_SRC test_e2e_tail_brc_tail_reduce_expect_kernel.cpp)
