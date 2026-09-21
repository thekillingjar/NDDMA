add_codegen_e2e_st_test(load_where_x2x3_is_ubscalar_store_expect_code
    TILING load_where_x2x3_is_ubscalar_store_codegen_tiling.cpp
    CODEGEN load_where_x2x3_is_ubscalar_store_codegen.cpp 
    KERNEL_SRC
        load_where_x2x3_is_ubscalar_store_kernel.cpp
        load_where_x2x3_is_ubscalar_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_where_x2x3_is_ubscalar_store_expect_kernel.cpp)
