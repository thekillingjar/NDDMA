# DumpFormat<a name="ZH-CN_TOPIC_0000002484471490"></a>

Dump文件的格式，头文件位于CANN软件安装后文件存储路径下的include/graph/graph.h。

```
enum class DumpFormat : uint32_t {
  kOnnx,      // 基于ONNX的模型描述结构，可以使用Netron等可视化软件打开，生成文件名为ge_onnx*.pbtxt
  kTxt,       // protobuf格式存储的文本文件，生成文件名为ge_proto*.txt
  kReadable   // 参考Dynamo风格的高可读性文本文件，生成文件名为ge_readable*.txt
};
```

