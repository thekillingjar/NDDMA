# aclTransType<a name="ZH-CN_TOPIC_0000001312400753"></a>

```
typedef enum aclTransType
{
    ACL_TRANS_N,    //不转置，默认为不转置
    ACL_TRANS_T,    //转置
    ACL_TRANS_NZ,   //内部格式，能够提供更好的接口性能，建议使用场景为：某个输入矩阵作为底库，执行一次转NZ格式的操作后，多次使用NZ格式的算子，提升性能。
    ACL_TRANS_NZ_T  //内部格式转置，当前不支持
}aclTransType;
```

