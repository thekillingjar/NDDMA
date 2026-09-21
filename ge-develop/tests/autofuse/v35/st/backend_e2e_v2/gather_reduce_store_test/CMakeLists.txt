backend_e2e_st_test(gather_reduce_store_test
        CODEGEN gather_reduce_store_backend_generate.cpp
        KERNEL_SRC
        test_e2e_gather_reduce_store_test_kernel.cpp
        test_e2e_gather_reduce_store_test_tiling.cpp
        autofuse_tiling_data.h
        TEST_SRC test_e2e_gather_reduce_store_expect_kernel.cpp)