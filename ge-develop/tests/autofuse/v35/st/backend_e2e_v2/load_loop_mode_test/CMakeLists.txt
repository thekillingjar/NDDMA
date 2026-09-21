backend_e2e_st_test(load_loop_mode_test
    CODEGEN load_loop_mode_backend_generate.cpp
    KERNEL_SRC
        load_loop_mode_test_kernel.cpp
        load_loop_mode_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_loop_mode_expect_kernel.cpp)
