add_codegen_e2e_st_test(pow_int32
    TILING pow_int32_codegen_tiling.cpp
    CODEGEN pow_int32_codegen.cpp
    KERNEL_SRC
        pow_int32_kernel.cpp
        pow_int32_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_pow_int32_expect_kernel.cpp)
