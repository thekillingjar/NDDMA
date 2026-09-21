# aclmdlInputAippType<a name="ZH-CN_TOPIC_0000001312641593"></a>

```
typedef enum {
    ACL_DATA_WITHOUT_AIPP = 0,  //该输入无AIPP信息
    ACL_DATA_WITH_STATIC_AIPP,  //该输入带静态AIPP信息
    ACL_DATA_WITH_DYNAMIC_AIPP, //该输入有关联的动态AIPP输入
    ACL_DYNAMIC_AIPP_NODE       //该输入本身就是动态AIPP输入，用于配置动态AIPP参数
} aclmdlInputAippType;
```

