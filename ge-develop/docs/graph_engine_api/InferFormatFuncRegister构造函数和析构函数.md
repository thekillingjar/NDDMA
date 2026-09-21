# InferFormatFuncRegister构造函数和析构函数<a name="ZH-CN_TOPIC_0000002531200353"></a>

## 头文件<a name="section6743111793111"></a>

\#include <graph/operator\_factory.h\>

## 功能说明<a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section36893359"></a>

InferFormatFuncRegister构造函数和析构函数。

## 函数原型<a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section136951948195410"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
InferFormatFuncRegister(const std::string &operator_type, const InferFormatFunc &infer_format_func)
InferFormatFuncRegister(const char_t *const operator_type, const InferFormatFunc &infer_format_func)
~InferFormatFuncRegister() = default
```

## 参数说明<a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section63604780"></a>

<a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="29.220000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="43.15%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a>operator_type</p>
</td>
<td class="cellrowborder" valign="top" width="29.220000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="43.15%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a>算子类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_row5981141282118"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312484749_p1942811810338"><a name="zh-cn_topic_0000001312484749_p1942811810338"></a><a name="zh-cn_topic_0000001312484749_p1942811810338"></a>infer_format_func</p>
</td>
<td class="cellrowborder" valign="top" width="29.220000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_p6982161222114"><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_p6982161222114"></a><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_p6982161222114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="43.15%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_p2982141292119"><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_p2982141292119"></a><a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_p2982141292119"></a>算子InferFormat函数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section35572112"></a>

InferFormatFuncRegister构造函数返回InferFormatFuncRegister类型的对象。

## 约束说明<a name="zh-cn_topic_0000001312484749_zh-cn_topic_0204328224_zh-cn_topic_0182636384_section62768825"></a>

算子InferFormat函数注册接口，此接口被其他头文件引用，一般不由算子开发者直接调用。

