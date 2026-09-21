add_codegen_e2e_st_test(load_nan_out_for_store_expect_code
    TILING load_nan_out_for_store_codegen_tiling.cpp
    CODEGEN load_nan_out_for_store_codegen.cpp
    KERNEL_SRC
        load_nan_out_for_store_kernel.cpp
        load_nan_out_for_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_nan_out_for_store_expect_kernel.cpp)
