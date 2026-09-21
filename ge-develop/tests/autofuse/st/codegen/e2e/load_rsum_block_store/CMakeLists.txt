add_codegen_e2e_st_test(load_rsum_block_store
    TILING load_rsum_block_store_codegen_tiling.cpp
    CODEGEN load_rsum_block_store_codegen.cpp
    KERNEL_SRC
        load_rsum_block_store_kernel.cpp
        load_rsum_block_store_tiling.cpp
        load_rsum_block_store_float_kernel.cpp
        load_rsum_block_store_float_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_rsum_block_store_kernel.cpp)
