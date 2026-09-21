backend_e2e_st_test(abs_clip_by_value_test
    CODEGEN abs_clip_by_value_backend_generate.cpp 
    KERNEL_SRC
        abs_clip_by_value_test_kernel.cpp
        abs_clip_by_value_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_abs_clip_by_value_expect_kernel.cpp)
