backend_e2e_st_test(concat_test
    CODEGEN concat_backend_generate.cpp
    KERNEL_SRC
        concat_test_kernel.cpp
        concat_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_concat_expect_kernel.cpp)
