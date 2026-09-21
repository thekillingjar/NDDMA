backend_e2e_st_test(concat_to_stores_test
    CODEGEN concat_to_stores_backend_generator.cpp
    KERNEL_SRC
        concat_to_stores_test_kernel.cpp
        concat_to_stores_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_concat_to_stores_kernel.cpp)
