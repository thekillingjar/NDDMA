backend_e2e_st_test(cos_bf16_test
    CODEGEN cos_bf16_backend_generate.cpp
    KERNEL_SRC
        cos_bf16_test_kernel.cpp
        cos_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_cos_bf16_expect_kernel.cpp)
