backend_e2e_st_test(axpy_abs_half_test
    CODEGEN axpy_abs_half_backend_generate.cpp 
    KERNEL_SRC
        axpy_abs_half_test_kernel.cpp
        axpy_abs_half_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_axpy_abs_half_expect_kernel.cpp)