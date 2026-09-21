add_codegen_e2e_st_test(load_ub2ub_abs_store_expect_code
    TILING load_ub2ub_abs_store_codegen_tiling.cpp
    CODEGEN load_ub2ub_abs_store_codegen.cpp 
    KERNEL_SRC
        load_ub2ub_abs_store_kernel.cpp
        load_ub2ub_abs_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_ub2ub_abs_store_expect_kernel.cpp)
