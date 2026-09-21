# HiddenInputSubType<a name="ZH-CN_TOPIC_0000002516551451"></a>

ArgDescInfo隐藏输入地址的类型，头文件位于CANN软件安装后文件存储路径下的include/graph/arg\_desc\_info.h。

```
enum class HiddenInputSubType {
  kHcom, // 用于通信的hiddenInput，MC2算子使用
  kEnd
};
```

