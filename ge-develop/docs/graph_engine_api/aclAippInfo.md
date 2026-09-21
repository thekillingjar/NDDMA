# aclAippInfo<a name="ZH-CN_TOPIC_0000001265081470"></a>

```
typedef struct aclAippInfo {
    aclAippInputFormat inputFormat;
    int32_t srcImageSizeW;
    int32_t srcImageSizeH;
    int8_t cropSwitch;
    int32_t loadStartPosW;
    int32_t loadStartPosH;
    int32_t cropSizeW;
    int32_t cropSizeH;
    int8_t resizeSwitch;
    int32_t resizeOutputW;
    int32_t resizeOutputH;
    int8_t paddingSwitch;
    int32_t leftPaddingSize;
    int32_t rightPaddingSize;
    int32_t topPaddingSize;
    int32_t bottomPaddingSize;
    int8_t cscSwitch;
    int8_t rbuvSwapSwitch;
    int8_t axSwapSwitch;
    int8_t singleLineMode;
    int32_t matrixR0C0;
    int32_t matrixR0C1;
    int32_t matrixR0C2;
    int32_t matrixR1C0;
    int32_t matrixR1C1;
    int32_t matrixR1C2;
    int32_t matrixR2C0;
    int32_t matrixR2C1;
    int32_t matrixR2C2;
    int32_t outputBias0;
    int32_t outputBias1;
    int32_t outputBias2;
    int32_t inputBias0;
    int32_t inputBias1;
    int32_t inputBias2;
    int32_t meanChn0;
    int32_t meanChn1;
    int32_t meanChn2;
    int32_t meanChn3;
    float minChn0;
    float minChn1;
    float minChn2;
    float minChn3;
    float varReciChn0;
    float varReciChn1;
    float varReciChn2;
    float varReciChn3;
    aclFormat srcFormat;      // 模型转换前，原始模型的输入format
    aclDataType srcDatatype;  // 模型转换前，原始模型的输入datatype
    size_t srcDimNum;         // 模型转换前，原始模型输入dims
    size_t shapeCount;        // 动态Shape（动态Batch或动态分辨率）场景下的档位数
    aclAippDims outDims[ACL_MAX_SHAPE_COUNT]; // 在多shape模型下，各个分档对应的aipp输出dims信息，需要在模型编译阶段按照shape推导出来，保存在模型中，输出的Format，datatype、输出dims等不随shape变化而变化
    aclAippExtendInfo *aippExtend; // 预留参数，当前版本用户必须传入空指针
} aclAippInfo;
```

