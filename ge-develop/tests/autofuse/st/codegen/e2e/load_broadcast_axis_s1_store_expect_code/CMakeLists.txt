add_codegen_e2e_st_test(load_broadcast_axis_s1_store_expect_code
    TILING load_broadcast_axis_s1_store_codegen_tiling.cpp
    CODEGEN load_broadcast_axis_s1_store_codegen.cpp 
    KERNEL_SRC
        load_broadcast_axis_s1_store_kernel.cpp
        load_broadcast_axis_s1_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_load_broadcast_axis_s1_store_codegen.cpp)
