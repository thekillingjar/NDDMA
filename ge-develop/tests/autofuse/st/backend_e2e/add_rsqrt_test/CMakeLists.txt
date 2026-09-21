backend_e2e_st_test(add_rsqrt_test
    CODEGEN add_rsqrt_backend_generator.cpp
    KERNEL_SRC
        add_rsqrt_test_kernel.cpp
        add_rsqrt_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_add_rsqrt_kernel.cpp)
