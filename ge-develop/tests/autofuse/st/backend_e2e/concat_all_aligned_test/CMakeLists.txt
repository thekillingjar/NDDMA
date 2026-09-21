backend_e2e_st_test(concat_all_aligned_test
    CODEGEN concat_all_aligned_backend_generator.cpp
    KERNEL_SRC
        concat_all_aligned_test_kernel.cpp
        concat_all_aligned_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_concat_all_aligned_kernel.cpp)
