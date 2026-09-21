backend_e2e_st_test(cast_reciprocal_test
    CODEGEN cast_reciprocal_backend_generate.cpp
    KERNEL_SRC
        cast_reciprocal_test_kernel.cpp
        cast_reciprocal_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_cast_reciprocal_expect_kernel.cpp)
