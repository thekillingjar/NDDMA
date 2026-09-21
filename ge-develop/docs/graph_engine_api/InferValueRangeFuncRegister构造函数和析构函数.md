# InferValueRangeFuncRegister构造函数和析构函数<a name="ZH-CN_TOPIC_0000002531200355"></a>

## 头文件<a name="section18580257113011"></a>

\#include <graph/operator\_factory.h\>

## 功能说明<a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section36893359"></a>

InferValueRangeFuncRegister构造函数和析构函数。

## 函数原型<a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section136951948195410"></a>

```
InferValueRangeFuncRegister(const char_t *const operator_type, const WHEN_CALL when_call,
const InferValueRangeFunc &infer_value_range_func)
InferValueRangeFuncRegister(const char_t *const operator_type)
~InferValueRangeFuncRegister() = default
```

## 参数说明<a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section63604780"></a>

<a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="28.84%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="43.53%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a>operator_type</p>
</td>
<td class="cellrowborder" valign="top" width="28.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="43.53%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a>算子类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_row5981141282118"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001601752412_p13230155092815"><a name="zh-cn_topic_0000001601752412_p13230155092815"></a><a name="zh-cn_topic_0000001601752412_p13230155092815"></a>when_call</p>
</td>
<td class="cellrowborder" valign="top" width="28.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_p6982161222114"><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_p6982161222114"></a><a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_p6982161222114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="43.53%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001601752412_p47631661757"><a name="zh-cn_topic_0000001601752412_p47631661757"></a><a name="zh-cn_topic_0000001601752412_p47631661757"></a>infer函数的调用场景。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001601752412_row188411547289"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001601752412_p2851054152811"><a name="zh-cn_topic_0000001601752412_p2851054152811"></a><a name="zh-cn_topic_0000001601752412_p2851054152811"></a>infer_value_range_func</p>
</td>
<td class="cellrowborder" valign="top" width="28.84%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001601752412_p1885454122819"><a name="zh-cn_topic_0000001601752412_p1885454122819"></a><a name="zh-cn_topic_0000001601752412_p1885454122819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="43.53%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001601752412_p78575411285"><a name="zh-cn_topic_0000001601752412_p78575411285"></a><a name="zh-cn_topic_0000001601752412_p78575411285"></a>算子InferValueRange函数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section35572112"></a>

InferValueRangeFuncRegister构造函数返回InferValueRangeFuncRegister类型的对象。

## 约束说明<a name="zh-cn_topic_0000001601752412_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section62768825"></a>

算子InferValueRangeFuncRegister函数注册接口，此接口被其他头文件引用，一般不由算子开发者直接调用。

