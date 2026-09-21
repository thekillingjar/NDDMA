add_codegen_e2e_st_test(load_compare_int64_tensor_ne_store
    TILING load_compare_int64_tensor_ne_store_codegen_tiling.cpp
    CODEGEN load_compare_int64_tensor_ne_store_codegen.cpp
    KERNEL_SRC
        load_compare_int64_tensor_ne_store_kernel.cpp
        load_compare_int64_tensor_ne_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_compare_int64_tensor_ne_store_expect_kernel.cpp)
