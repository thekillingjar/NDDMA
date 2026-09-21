add_codegen_e2e_st_test(load_rprod_store
    TILING load_rprod_store_codegen_tiling.cpp
    CODEGEN load_rprod_store_codegen.cpp
    KERNEL_SRC
        load_rprod_store_kernel.cpp
        load_rprod_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_rprod_store_kernel.cpp)

