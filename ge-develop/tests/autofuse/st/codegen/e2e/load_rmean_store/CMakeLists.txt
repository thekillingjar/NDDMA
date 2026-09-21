add_codegen_e2e_st_test(load_rmean_store
    TILING load_rmean_store_codegen_tiling.cpp
    CODEGEN load_rmean_store_codegen.cpp
    KERNEL_SRC
        load_rmean_store_kernel.cpp
        load_rmean_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_rmean_store_kernel.cpp)

