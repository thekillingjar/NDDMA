backend_e2e_st_test(split_test
    CODEGEN split_backend_generate.cpp
    KERNEL_SRC
        split_test_kernel.cpp
        split_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_split_expect_kernel.cpp)
