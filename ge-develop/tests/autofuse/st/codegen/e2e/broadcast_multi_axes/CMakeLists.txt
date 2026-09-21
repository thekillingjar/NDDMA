add_codegen_e2e_st_test(broadcast_multi_axes
    TILING broadcast_multi_axes_codegen_tiling.cpp
    CODEGEN broadcast_multi_axes_codegen.cpp 
    KERNEL_SRC
        broadcast_multi_axes_kernel.cpp
        broadcast_multi_axes_tiling.cpp
        autofuse_tiling_data.h
        broadcast_multi_axes_kernel_bab.cpp
    TEST_SRC test_e2e_broadcast_multi_axes_kernel.cpp)
