add_codegen_e2e_st_test(schedule_multi_group_ws_reuse_output
    TILING schedule_multi_group_ws_reuse_output_codegen_tiling.cpp
    CODEGEN schedule_multi_group_ws_reuse_output_codegen.cpp 
    KERNEL_SRC
        schedule_multi_group_ws_reuse_output_kernel.cpp
        schedule_multi_group_ws_reuse_output_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_schedule_multi_group_ws_reuse_output.cpp)

