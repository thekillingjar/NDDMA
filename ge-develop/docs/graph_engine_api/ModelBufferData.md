# ModelBufferData<a name="ZH-CN_TOPIC_0000001265245830"></a>

内存缓冲区中的序列化模型数据，头文件位于CANN软件安装后文件存储路径下的include/ge/ge\_ir\_build.h。

```
struct ModelBufferData
{
  std::shared_ptr<uint8_t> data = nullptr;
  uint64_t length;
};
```

其中data指向生成的模型数据，length代表该模型的实际大小。

