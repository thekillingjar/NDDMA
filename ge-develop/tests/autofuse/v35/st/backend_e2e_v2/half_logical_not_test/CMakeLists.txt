backend_e2e_st_test(half_logical_not_test
    CODEGEN half_logical_not_backend_generate.cpp
    KERNEL_SRC
        half_logical_not_test_kernel.cpp
        half_logical_not_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_half_logical_not_expect_kernel.cpp)