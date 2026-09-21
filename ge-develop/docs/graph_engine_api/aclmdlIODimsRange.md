# aclmdlIODimsRange<a name="ZH-CN_TOPIC_0000002029129665"></a>

```
#define ACL_DIM_ENDPOINTS        2
#define ACL_MAX_DIM_CNT          128

typedef struct aclmdlIODimsRange {
    size_t rangeCount; /** 维度数量 */
    int64_t range[ACL_MAX_DIM_CNT][ACL_DIM_ENDPOINTS]; /** Shape范围二维数组，第一维表示维度数量，第二维表示范围的上、下限 */
} aclmdlIODimsRange;
```

