# InputTensorInfo<a name="ZH-CN_TOPIC_0000001265085946"></a>

输入Tensor信息，头文件位于CANN软件安装后文件存储路径下的include/ge/ge\_api\_types.h。

```
struct InputTensorInfo {
  uint32_t data_type;               // data type
  std::vector<int64_t> dims;        // shape description
  void *data;                       // tensor data
  int64_t length;                   // tensor length
};
```

