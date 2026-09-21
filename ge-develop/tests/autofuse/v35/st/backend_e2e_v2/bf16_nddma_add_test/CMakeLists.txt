backend_e2e_st_test(bf16_nddma_add_test
    CODEGEN bf16_nddma_add_backend_generate.cpp 
    KERNEL_SRC
        bf16_nddma_add_test_kernel.cpp
        bf16_nddma_add_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_bf16_nddma_add_expect_kernel.cpp)
