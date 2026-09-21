# MemoryType<a name="ZH-CN_TOPIC_0000001844910226"></a>

内存类型枚举，头文件位于CANN软件安装后文件存储路径下的include/ge/ge\_api\_types.h。

```
enum class MemoryType : std::int64_t {
  /*
   * call aclrtMalloc with aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST,
   * ACL_MEM_MALLOC_HUGE_ONLY, ACL_MEM_MALLOC_NORMAL_ONLY
   */
  MEMORY_TYPE_DEFAULT,
  /*
   * call aclrtMalloc with aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST_P2P,
   * ACL_MEM_MALLOC_HUGE_ONLY_P2P, ACL_MEM_MALLOC_NORMAL_ONLY_P2P
   */
  MEMORY_TYPE_P2P
};
```

各枚举项说明如下：

-   MEMORY\_TYPE\_DEFAULT：默认Device内存类型，可以通过调用“aclrtMalloc”接口申请得到，申请内存时传入“aclrtMemMallocPolicy”中的3种枚举值之一，ACL\_MEM\_MALLOC\_HUGE\_FIRST、ACL\_MEM\_MALLOC\_HUGE\_ONLY、 ACL\_MEM\_MALLOC\_NORMAL\_ONLY
-   MEMORY\_TYPE\_P2P：仅Device之间内存复制场景下的内存类型，可以通过调用aclrtMalloc接口申请得到，申请内存时传入aclrtMemMallocPolicy中的3种枚举值之一，ACL\_MEM\_MALLOC\_HUGE\_FIRST\_P2P、ACL\_MEM\_MALLOC\_HUGE\_ONLY\_P2P、ACL\_MEM\_MALLOC\_NORMAL\_ONLY\_P2P

aclrtMalloc、aclrtMemMallocPolicy接口说明请参见《应用开发指南 \(C&C++\)》。

