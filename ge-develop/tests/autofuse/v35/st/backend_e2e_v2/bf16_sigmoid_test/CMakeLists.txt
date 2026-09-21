backend_e2e_st_test(bf16_sigmoid_test
    CODEGEN bf16_sigmoid_backend_generator.cpp
    KERNEL_SRC
        bf16_sigmoid_test_kernel.cpp
        bf16_sigmoid_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_bf16_sigmoid_kernel.cpp)