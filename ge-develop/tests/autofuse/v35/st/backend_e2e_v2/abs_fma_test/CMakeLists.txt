backend_e2e_st_test(abs_fma_test
    CODEGEN abs_fma_backend_generate.cpp 
    KERNEL_SRC
        abs_fma_test_kernel.cpp
        abs_fma_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_abs_fma_expect_kernel.cpp)
