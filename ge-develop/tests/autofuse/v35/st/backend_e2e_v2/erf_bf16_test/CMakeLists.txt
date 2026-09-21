backend_e2e_st_test(erf_bf16_test
    CODEGEN erf_bf16_backend_generate.cpp
    KERNEL_SRC
        erf_bf16_test_kernel.cpp
        erf_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_erf_bf16_expect_kernel.cpp)
