# InferenceContext构造函数和析构函数<a name="ZH-CN_TOPIC_0000002531200345"></a>

## 头文件<a name="section6743111793111"></a>

\#include <graph/inference\_context.h\>

## 功能说明<a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section36893359"></a>

InferenceContext对象的构造函数和析构函数。

## 函数原型<a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section136951948195410"></a>

```
~InferenceContext() = default
InferenceContext(const InferenceContext &context) = delete
InferenceContext(const InferenceContext &&context) = delete
InferenceContext &operator=(const InferenceContext &context) = delete
InferenceContext &operator=(const InferenceContext &&context) = delete
```

## 参数说明<a name="zh-cn_topic_0000001312484729_section144401754174018"></a>

<a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="18.029999999999998%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.399999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.57%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="18.029999999999998%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a>context</p>
</td>
<td class="cellrowborder" valign="top" width="14.399999999999999%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.57%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a>InferenceContext内容，供初始化使用。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section35572112"></a>

InferenceContext构造函数返回InferenceContext类型的对象。

## 异常处理<a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section51713556"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001312484729_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section62768825"></a>

无。

