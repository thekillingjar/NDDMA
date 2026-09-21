add_codegen_e2e_st_test(load_isfinite_store
    TILING load_isfinite_store_codegen_tiling.cpp
    CODEGEN load_isfinite_store_codegen.cpp 
    KERNEL_SRC
        load_isfinite_store_kernel.cpp
        load_isfinite_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_isfinite_store_expect_kernel.cpp)
