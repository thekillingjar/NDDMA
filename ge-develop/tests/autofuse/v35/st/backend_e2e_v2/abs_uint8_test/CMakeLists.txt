backend_e2e_st_test(abs_uint8_test
    CODEGEN abs_uint8_backend_generate.cpp
    KERNEL_SRC
        abs_uint8_test_kernel.cpp
        abs_uint8_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_abs_uint8_expect_kernel.cpp)
