add_codegen_e2e_st_test(load_rsum_ra_store
    TILING load_rsum_ra_store_codegen_tiling.cpp
    CODEGEN load_rsum_ra_store_codegen.cpp
    KERNEL_SRC
        load_rsum_ra_store_kernel.cpp
        load_rsum_ra_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_rsum_ra_store_kernel.cpp)
