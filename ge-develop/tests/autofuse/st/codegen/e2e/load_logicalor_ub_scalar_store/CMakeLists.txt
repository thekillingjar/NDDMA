add_codegen_e2e_st_test(load_logicalor_ub_scalar_store
    TILING load_logicalor_ub_scalar_store_codegen_tiling.cpp
    CODEGEN load_logicalor_ub_scalar_store_codegen.cpp
    KERNEL_SRC
        load_logicalor_ub_scalar_store_kernel.cpp
        load_logicalor_ub_scalar_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_logicalor_ub_scalar_store_expect_kernel.cpp)
