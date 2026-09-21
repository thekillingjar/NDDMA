backend_e2e_st_test(add_gelu_test
    CODEGEN add_gelu_backend_generate.cpp 
    KERNEL_SRC
        add_gelu_test_kernel.cpp
        add_gelu_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_add_gelu_expect_kernel.cpp)
