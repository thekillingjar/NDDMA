add_codegen_e2e_st_test(load_erf_store
    TILING load_erf_store_codegen_tiling.cpp
    CODEGEN load_erf_store_codegen.cpp 
    KERNEL_SRC
        load_erf_store_kernel.cpp
        load_erf_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_erf_store_kernel.cpp)
