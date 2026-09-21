backend_e2e_st_test(bf16_sin_test
    CODEGEN bf16_sin_backend_generator.cpp
    KERNEL_SRC
        bf16_sin_test_kernel.cpp
        bf16_sin_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_bf16_sin_kernel.cpp)