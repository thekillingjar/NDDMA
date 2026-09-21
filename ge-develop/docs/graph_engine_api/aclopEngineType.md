# aclopEngineType<a name="ZH-CN_TOPIC_0000001312721421"></a>

```
typedef enum aclEngineType {
   ACL_ENGINE_SYS,      // 不关心具体执行引擎时填写，当前仅支持设置为该选项
   ACL_ENGINE_AICORE,   // 将算子编译成AI Core算子，当前不支持
   ACL_ENGINE_VECTOR   //  将算子编译成Vector Core算子，当前不支持
} aclopEngineType;
```

