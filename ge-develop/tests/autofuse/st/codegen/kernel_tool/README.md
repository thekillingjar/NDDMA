一、工具用途
该工具主要使用CPU仿真方式定位kernel存在的UB非对齐访问类问题

二、工具文件列表及说明
- 1 autofuse_tiling_data.h
>拷贝kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_tiling_data.h的内容
- 2 autofuse_tiling_func_common.h
>拷贝kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_tiling_func_common.h的内容，并将下列宏定义修改为空
```cpp
#define OP_LOGD(name, fmt, ...) 
#define OP_LOGI(name, fmt, ...) 
#define OP_LOGW(name, fmt, ...) 
#define OP_LOGE(name, fmt, ...) 
#define OP_EVENT(name, fmt, ...)
```
- 3 kernel.cpp 
>拷贝kernel_meta_xxxxxxx/te_ascbackend_xxxx/device/autofuse_xxx_op_kernel.cpp的内容，并在文件头添加以下两行代码

```cpp
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
```
- 4 main.cpp
>主函数，用于构造输入、真值、tiling data,并校验kernel计算结果与真值，具体参考其实现
- 5 Makefile
>构建工程脚本
- 6 README.md
>使用说明
- 7 tiling_func_asc_graph0_schedule_result0_g0.cpp
>拷贝kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_xxx_tiling_func_asc_graph0_schedule_result0_g0.cpp的内容
- 8 tiling_func_schedule_group_tail.cpp
>拷贝kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_xxx_tiling_func_schedule_group_tail.cpp的内容
- 9 tiling_func_solver_func.cpp
>拷贝kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_xxx_tiling_func_solver_func.cpp的内容
- 10 tiling_func_tiling_def_and_tiling_const.cpp
>拷贝kernel_meta_xxxxxxx/te_ascbackend_xxxx/host/autofuse_xxx_tiling_func_tiling_def_and_tiling_const.cpp的内容


三、参考main.cpp实现测试用例
main.cpp是测试用例入口，需要参考其实现，构造对应的实现，当前只支持动态shape场景，如果是静态shape，可以修改为动态shape。


四、编译
```sh
make CANN_INSTALL_PATH=/usr/local/Ascend/latest #替换为自己的cann包路径
```

五、执行工具
```sh
source /usr/local/Ascend/latest/bin/setenv.bash #设置cann包环境变量
./test_kernel
```