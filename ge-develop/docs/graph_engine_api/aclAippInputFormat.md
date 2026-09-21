# aclAippInputFormat<a name="ZH-CN_TOPIC_0000001265241326"></a>

静态AIPP和动态AIPP支持的图片格式不同，详情如下：

```
typedef enum 
{
    ACL_YUV420SP_U8 = 1,    // YUV420SP_U8，静态AIPP和动态AIPP都支持
    ACL_XRGB8888_U8=2,      // XRGB8888_U8，静态AIPP和动态AIPP都支持
    ACL_RGB888_U8=3,        // RGB888_U8，静态AIPP和动态AIPP都支持
    ACL_YUV400_U8=4,        // YUV400_U8，静态AIPP和动态AIPP都支持
    ACL_NC1HWC0DI_FP16 = 5, // 暂不支持
    ACL_NC1HWC0DI_S8 = 6,   // 暂不支持
    ACL_ARGB8888_U8 = 7,    // 暂不支持 
    ACL_YUYV_U8 = 8,        // 暂不支持
    ACL_YUV422SP_U8 = 9,    // 暂不支持
    ACL_AYUV444_U8 = 10,    // 暂不支持
    ACL_RAW10 = 11,         // 暂不支持
    ACL_RAW12 = 12,         // 暂不支持
    ACL_RAW16 = 13,         // 暂不支持
    ACL_RAW24 = 14,         // 暂不支持
    ACL_AIPP_RESERVED = 0xffff,
} aclAippInputFormat;
```

