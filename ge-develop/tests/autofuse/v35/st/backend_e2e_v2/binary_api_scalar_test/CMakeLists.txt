backend_e2e_st_test(binary_api_scalar_test
    CODEGEN binary_api_scalar_backend_generate.cpp 
    KERNEL_SRC
        binary_api_scalar_test_kernel.cpp
        binary_api_scalar_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_binary_api_scalar_expect_kernel.cpp)
