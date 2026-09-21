add_codegen_e2e_st_test(load_rsum_store
    TILING load_rsum_store_codegen_tiling.cpp
    CODEGEN load_rsum_store_codegen.cpp
    KERNEL_SRC
        load_rsum_store_kernel.cpp
        load_rsum_store_tiling.cpp
        autofuse_tiling_data.h
        load_rsum_store_kernel_int32.cpp
        load_rsum_store_tiling_int32.cpp
    TEST_SRC test_load_rsum_store_kernel.cpp)
