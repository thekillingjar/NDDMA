# OutputTensorInfo<a name="ZH-CN_TOPIC_0000001264926014"></a>

输出Tensor信息，头文件位于CANN软件安装后文件存储路径下的include/ge/ge\_api\_types.h。

```
struct OutputTensorInfo {
  uint32_t data_type{};             // data type
  std::vector<int64_t> dims;        // shape description
  std::unique_ptr<uint8_t[]> data;  // tensor data
  int64_t length{};                 // tensor length
  OutputTensorInfo() : dims({}), data(nullptr) {}
  OutputTensorInfo(OutputTensorInfo &&out) noexcept = default;
  OutputTensorInfo &operator=(OutputTensorInfo &&out)& noexcept {
    if (this != &out) {
      data_type = out.data_type;
      dims = out.dims;
      data = std::move(out.data);
      length = out.length;
    }
    return *this;
  }
  OutputTensorInfo(const OutputTensorInfo &) = delete;
  OutputTensorInfo &operator=(const OutputTensorInfo &)& = delete;
};
```

