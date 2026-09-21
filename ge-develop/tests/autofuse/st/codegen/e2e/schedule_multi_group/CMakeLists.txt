add_codegen_e2e_st_test(schedule_multi_group
    TILING schedule_multi_group_codegen_tiling.cpp
    CODEGEN schedule_multi_group_codegen.cpp 
    KERNEL_SRC
        schedule_multi_group_kernel.cpp
        schedule_multi_group_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_schedule_multi_group.cpp)

