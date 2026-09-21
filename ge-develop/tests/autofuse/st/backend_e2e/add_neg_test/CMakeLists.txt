backend_e2e_st_test(add_neg_test
    CODEGEN add_neg_backend_generate.cpp 
    KERNEL_SRC
        add_neg_test_kernel.cpp
        add_neg_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_add_neg_expect_kernel.cpp)
