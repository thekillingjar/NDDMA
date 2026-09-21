add_codegen_e2e_st_test(load_rmin_store
    TILING load_rmin_store_codegen_tiling.cpp
    CODEGEN load_rmin_store_codegen.cpp
    KERNEL_SRC
        load_rmin_store_kernel.cpp
        load_rmin_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_rmin_store_kernel.cpp)

