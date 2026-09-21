# Operator构造函数和析构函数<a name="ZH-CN_TOPIC_0000002499360460"></a>

## 头文件和库文件<a name="section6743111793111"></a>

-   头文件：\#include <graph/operator.h\>
-   库文件：libgraph.so

## 功能说明<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636384_section36893359"></a>

Operator构造函数和析构函数。

## 函数原型<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636384_section136951948195410"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
Operator()
explicit Operator(const std::string &type)
explicit Operator(const char_t *type)
Operator(const std::string &name, const std::string &type)
Operator(const AscendString &name, const AscendString &type)
Operator(const char_t *name, const char_t *type)
virtual ~Operator() = default
```

## 参数说明<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636384_section63604780"></a>

<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.66%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="47.71%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="24.66%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.71%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a>算子类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0204328224_row5981141282118"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0204328224_p69811812192113"><a name="zh-cn_topic_0204328224_p69811812192113"></a><a name="zh-cn_topic_0204328224_p69811812192113"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="24.66%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0204328224_p6982161222114"><a name="zh-cn_topic_0204328224_p6982161222114"></a><a name="zh-cn_topic_0204328224_p6982161222114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.71%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0204328224_p2982141292119"><a name="zh-cn_topic_0204328224_p2982141292119"></a><a name="zh-cn_topic_0204328224_p2982141292119"></a>算子名称。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636384_section35572112"></a>

Operator构造函数返回Operator类型的对象。

## 异常处理<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636384_section51713556"></a>

无。

## 约束说明<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636384_section62768825"></a>

无。

