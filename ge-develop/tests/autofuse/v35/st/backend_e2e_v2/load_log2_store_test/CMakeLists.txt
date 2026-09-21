backend_e2e_st_test(load_log2_store_test
    CODEGEN load_log2_store_backend_generator.cpp
    KERNEL_SRC
        load_log2_store_test_kernel.cpp
        load_log2_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_log2_store_kernel.cpp)
