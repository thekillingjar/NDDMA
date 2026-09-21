# ProfilingAicoreMetrics<a name="ZH-CN_TOPIC_0000001312485901"></a>

Profiling AI Core metrics，头文件位于CANN软件安装后文件存储路径下的include/ge/ge\_prof.h。

```
enum ProfilingAicoreMetrics {
  kAicoreArithmeticUtilization = 0,  // 各种计算类指标占比统计
  kAicorePipeUtilization = 1,  // 计算单元和搬运单元耗时占比
  kAicoreMemory = 2,  // UB/L1/L2读写带宽数据
  kAicoreMemoryL0 = 3,  // L0读写带宽数据
  kAicoreResourceConflictRatio = 4,  // 流水线队列类指令占比
  kAicoreMemoryUB = 5,   // 细粒度的UB读写带宽数据
  kAicoreL2Cache = 6,   //读写cache命中次数和缺失后重新分配次数，支持的产品：
                        // Atlas A2 训练系列产品/Atlas A2 推理系列产品
                        // Atlas A3 训练系列产品/Atlas A3 推理系列产品
};
```

