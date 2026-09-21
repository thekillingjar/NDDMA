backend_e2e_st_test(compare_test
    CODEGEN compare_backend_generate.cpp 
    KERNEL_SRC
        compare_test_kernel.cpp
        compare_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_compare_expect_kernel.cpp)
