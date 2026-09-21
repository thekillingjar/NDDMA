add_codegen_e2e_st_test(load_rmax_store
    TILING load_rmax_store_codegen_tiling.cpp
    CODEGEN load_rmax_store_codegen.cpp
    KERNEL_SRC
        load_rmax_store_kernel.cpp
        load_rmax_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_rmax_store_kernel.cpp)

