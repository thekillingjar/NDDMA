backend_e2e_st_test(pgo_add_abs_test
    CODEGEN pgo_add_abs_backend_generate.cpp 
    KERNEL_SRC
        pgo_add_abs_test_kernel.cpp
        autofuse_tiling_data.h
        autofuse_tiling_func_common.h
        pgo_add_abs_asc_graph0_schedule_result0_g0.cpp
        pgo_add_abs_schedule_group_tail.cpp
        pgo_add_abs_solver_func.cpp
        pgo_add_abs_tiling_def_and_tiling_const.cpp
    TEST_SRC test_e2e_pgo_add_abs_expect_kernel.cpp)
