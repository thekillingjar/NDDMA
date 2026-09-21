add_codegen_e2e_st_test(load_max_min_store
    TILING load_max_min_store_codegen_tiling.cpp
    CODEGEN load_max_min_store_codegen.cpp
    KERNEL_SRC
        load_max_min_store_kernel.cpp
        load_max_min_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_max_min_store_expect_kernel.cpp)
