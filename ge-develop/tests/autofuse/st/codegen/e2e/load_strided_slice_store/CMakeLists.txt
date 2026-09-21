add_codegen_e2e_st_test(load_strided_slice_store
    TILING load_strided_slice_store_codegen_tiling.cpp
    CODEGEN load_strided_slice_store_codegen.cpp
    KERNEL_SRC
        load_strided_slice_store_kernel.cpp
        load_strided_slice_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_strided_slice_store_kernel.cpp)