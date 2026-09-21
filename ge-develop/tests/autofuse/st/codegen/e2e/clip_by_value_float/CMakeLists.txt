add_codegen_e2e_st_test(clip_by_value_float
    TILING clip_by_value_float_codegen_tiling.cpp
    CODEGEN clip_by_value_float_codegen.cpp
    KERNEL_SRC
        clip_by_value_float_kernel.cpp
        clip_by_value_float_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_clip_by_value_float_expect_kernel.cpp)
