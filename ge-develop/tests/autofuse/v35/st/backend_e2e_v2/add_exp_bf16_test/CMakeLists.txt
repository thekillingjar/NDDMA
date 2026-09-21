backend_e2e_st_test(add_exp_bf16_test
    CODEGEN add_exp_bf16_backend_generate.cpp 
    KERNEL_SRC
        add_exp_bf16_test_kernel.cpp
        add_exp_bf16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_add_exp_bf16_expect_kernel.cpp)
