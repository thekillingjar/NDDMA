backend_e2e_st_test(ceil_bf16_test
    CODEGEN ceil_bf16_backend_generate.cpp
    KERNEL_SRC
        ceil_bf16_test_kernel.cpp
        ceil_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_ceil_bf16_expect_kernel.cpp)
