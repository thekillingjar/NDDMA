backend_e2e_st_test(acos_bf16_test
    CODEGEN acos_bf16_backend_generator.cpp
    KERNEL_SRC
        acos_bf16_test_kernel.cpp
        acos_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_acos_bf16_expect_kernel.cpp)
