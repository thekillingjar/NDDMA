backend_e2e_st_test(load_gather_one_axis_split_b_t_abs_store_test
    CODEGEN load_gather_one_axis_abs_store_backend_generate.cpp
    KERNEL_SRC
        load_gather_one_axis_abs_store_test_kernel.cpp
        load_gather_one_axis_abs_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_gather_one_axis_b_t_abs_store_expect_kernel.cpp)
