backend_e2e_st_test(load_pow_all_input_is_scalar_store_test
        CODEGEN load_pow_all_input_is_scalar_store_backend_generate.cpp
        KERNEL_SRC
            load_pow_all_input_is_scalar_store_test_kernel.cpp
            load_pow_all_input_is_scalar_store_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_load_pow_all_input_is_scalar_store_expect_kernel.cpp)
