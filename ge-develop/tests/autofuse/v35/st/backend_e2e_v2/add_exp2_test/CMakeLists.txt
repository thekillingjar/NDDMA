backend_e2e_st_test(add_exp2_test
    CODEGEN add_exp2_backend_generate.cpp 
    KERNEL_SRC
        add_exp2_test_kernel.cpp
        add_exp2_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_add_exp2_expect_kernel.cpp)
