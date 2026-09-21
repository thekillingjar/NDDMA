backend_e2e_st_test(compare_x2_tensor_test
    CODEGEN compare_x2_tensor_backend_generate.cpp 
    KERNEL_SRC
        compare_x2_tensor_test_kernel.cpp
        compare_x2_tensor_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_compare_x2_tensor_expect_kernel.cpp)
