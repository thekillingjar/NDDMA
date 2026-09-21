add_codegen_e2e_st_test(gather_abs_one_axis
    TILING gather_abs_store_codegen_tiling.cpp
    CODEGEN gather_abs_store_codegen.cpp
    KERNEL_SRC
        gather_abs_store_kernel.cpp
        gather_abs_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_gather_abs_store_expect_kernel.cpp)