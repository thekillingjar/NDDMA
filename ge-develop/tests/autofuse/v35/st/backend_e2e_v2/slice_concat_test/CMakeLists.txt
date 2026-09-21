backend_e2e_st_test(slice_concat_test
    CODEGEN slice_concat_backend_generate.cpp
    KERNEL_SRC
        slice_concat_test_kernel.cpp
        slice_concat_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_slice_concat_expect_kernel.cpp)
