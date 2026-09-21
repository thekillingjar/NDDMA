add_codegen_e2e_st_test(load_cast_store_expect_code
    TILING load_cast_store_codegen_tiling.cpp
    CODEGEN load_cast_store_codegen.cpp
    KERNEL_SRC
        load_cast_store_kernel.cpp
        load_cast_store_tiling.cpp
        autofuse_tiling_data.h
        load_cast_store_uint8_kernel.cpp
    TEST_SRC test_e2e_load_cast_store_expect_kernel.cpp)
