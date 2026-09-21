add_codegen_e2e_st_test(load_multi_axis_store
    TILING load_multi_axis_store_codegen_tiling.cpp
    CODEGEN load_multi_axis_store_codegen.cpp
    KERNEL_SRC
        load_multi_axis_store_kernel.cpp
        load_multi_axis_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_multi_axis_store_kernel.cpp)

