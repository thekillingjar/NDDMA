backend_e2e_st_test(load_brc_test
        CODEGEN load_brc_backend_generate.cpp
        KERNEL_SRC
            load_brc_test_kernel.cpp
            load_brc_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_load_brc_expect_kernel.cpp)
