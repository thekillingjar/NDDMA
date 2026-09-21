add_codegen_e2e_st_test(store_empty_tensor        # 用于拼接生成第一二类st执行文件名字及so的名字
    TILING  store_empty_tensor_codegen_tiling.cpp  # 用于生成tiling so的源文件
    CODEGEN store_empty_tensor_codegen.cpp        # 第一类用例的原文件
    KERNEL_SRC                                           # 第一类用例执行生成的文件，同时也作为第二类用例的源文件
        store_empty_tensor_kernel.cpp
        store_empty_tensor_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_store_empty_tensor_kernel.cpp)   # 第二类用例的源文件