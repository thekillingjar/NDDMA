add_codegen_e2e_st_test(broadcast_force_merge
    TILING broadcast_force_merge_codegen_tiling.cpp
    CODEGEN broadcast_force_merge_codegen.cpp 
    KERNEL_SRC
        broadcast_force_merge_kernel.cpp
        broadcast_force_merge_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_broadcast_force_merge_kernel.cpp)
