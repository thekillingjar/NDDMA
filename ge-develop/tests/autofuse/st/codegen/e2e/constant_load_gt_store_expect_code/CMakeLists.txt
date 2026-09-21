# todo:下面用例依赖node支持通过graph copyfrom属性组，metadef支持该功能后再上线该用例
# 影响Constant, IndexExpr节点获取属性值
add_codegen_e2e_st_test(constant_load_gt_store_expect_code
    TILING constant_load_gt_store_codegen_tiling.cpp
    CODEGEN constant_load_gt_store_codegen.cpp
    KERNEL_SRC
        constant_load_gt_store_kernel.cpp
        constant_load_gt_store_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_constant_load_gt_store_expect_kernel.cpp)
