# aclmdlIODims<a name="ZH-CN_TOPIC_0000001312481769"></a>

```
#define ACL_MAX_DIM_CNT          128
#define ACL_MAX_TENSOR_NAME_LEN  128
typedef struct aclmdlIODims {
    char name[ACL_MAX_TENSOR_NAME_LEN]; // tensor name 
    size_t dimCount;                    // Shape中的维度个数，如果为标量，则维度个数为0
    int64_t dims[ACL_MAX_DIM_CNT];      // 维度信息
} aclmdlIODims;
```

