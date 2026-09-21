add_codegen_e2e_st_test(concat_mult_inputs        # 用于拼接生成第一二类st执行文件名字及so的名字
    TILING  concat_mult_inputs_codegen_tiling.cpp  # 用于生成tiling so的源文件
    CODEGEN concat_mult_inputs_codegen.cpp        # 第一类用例的原文件
    KERNEL_SRC                                           # 第一类用例执行生成的文件，同时也作为第二类用例的源文件
        concat_mult_inputs_kernel.cpp
        concat_mult_inputs_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_concat_mult_inputs_kernel.cpp)   # 第二类用例的源文件