add_codegen_e2e_st_test(load_gather_first_axis_split_b_t_abs_store
    TILING load_gather_abs_store_codegen_tiling.cpp
    CODEGEN load_gather_abs_store_codegen.cpp 
    KERNEL_SRC
        load_gather_abs_store_kernel.cpp
        load_gather_abs_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_gather_abs_store_codegen.cpp)
