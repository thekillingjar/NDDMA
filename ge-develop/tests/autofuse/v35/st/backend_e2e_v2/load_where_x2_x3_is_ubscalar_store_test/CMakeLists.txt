backend_e2e_st_test(load_where_x2_x3_is_ubscalar_store_test
        CODEGEN load_where_x2_x3_is_ubscalar_store_backend_generate.cpp
        KERNEL_SRC
            load_where_x2_x3_is_ubscalar_store_test_kernel.cpp
            load_where_x2_x3_is_ubscalar_store_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_load_where_x2_x3_is_ubscalar_store_expect_kernel.cpp)
