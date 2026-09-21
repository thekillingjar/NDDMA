# ProfDataTypeConfig<a name="ZH-CN_TOPIC_0000001312485865"></a>

Profiling数据类型配置文件，头文件位于CANN软件安装后文件存储路径下的include/ge/ge\_prof.h。

```
enum ProfDataTypeConfig {
  kProfTaskTime,          = 0x0002,      // 采集算子下发耗时、算子执行耗时数据以及算子基本信息数据，提供更全面的性能分析数据
  kProfAiCoreMetrics,     = 0x0004,      // AICore数据
  kProfAicpu,             = 0x0008,      // AI CPU数据，当前版本暂不支持
  kProfL2cache,            = 0x0010,      // L2 Cache采样数据
  kProfHccl,              = 0x0020,      // 集合通信数据
  kProfTrainingTrace,     = 0x0040,      // 迭代轨迹数据
  kProfTaskTimeL0,        = 0x0800,      // 采集算子下发耗时、算子执行耗时数据。与kProfTaskTime相比，由于不采集算子基本信息数据，采集时性能开销较小，可更精准统计相关耗时数据
};
```

