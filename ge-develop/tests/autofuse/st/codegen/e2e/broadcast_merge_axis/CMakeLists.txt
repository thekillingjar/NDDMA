add_codegen_e2e_st_test(broadcast_merge_axis
    TILING broadcast_merge_axis_codegen_tiling.cpp
    CODEGEN broadcast_merge_axis_codegen.cpp 
    KERNEL_SRC
        broadcast_merge_axis_kernel.cpp
        broadcast_merge_axis_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_broadcast_merge_axis.cpp)

