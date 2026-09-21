# GetValue<a name="ZH-CN_TOPIC_0000002531280391"></a>

## 产品支持情况<a name="section789110355111"></a>

<a name="zh-cn_topic_0000001312404881_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312404881_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001312404881_p1883113061818"><a name="zh-cn_topic_0000001312404881_p1883113061818"></a><a name="zh-cn_topic_0000001312404881_p1883113061818"></a><span id="zh-cn_topic_0000001312404881_ph20833205312295"><a name="zh-cn_topic_0000001312404881_ph20833205312295"></a><a name="zh-cn_topic_0000001312404881_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001312404881_p783113012187"><a name="zh-cn_topic_0000001312404881_p783113012187"></a><a name="zh-cn_topic_0000001312404881_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312404881_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312404881_p48327011813"><a name="zh-cn_topic_0000001312404881_p48327011813"></a><a name="zh-cn_topic_0000001312404881_p48327011813"></a><span id="zh-cn_topic_0000001312404881_ph583230201815"><a name="zh-cn_topic_0000001312404881_ph583230201815"></a><a name="zh-cn_topic_0000001312404881_ph583230201815"></a><term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312404881_p108715341013"><a name="zh-cn_topic_0000001312404881_p108715341013"></a><a name="zh-cn_topic_0000001312404881_p108715341013"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312404881_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312404881_p14832120181815"><a name="zh-cn_topic_0000001312404881_p14832120181815"></a><a name="zh-cn_topic_0000001312404881_p14832120181815"></a><span id="zh-cn_topic_0000001312404881_ph1483216010188"><a name="zh-cn_topic_0000001312404881_ph1483216010188"></a><a name="zh-cn_topic_0000001312404881_ph1483216010188"></a><term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312404881_p19948143911820"><a name="zh-cn_topic_0000001312404881_p19948143911820"></a><a name="zh-cn_topic_0000001312404881_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 头文件<a name="section6743111793111"></a>

-   头文件：\#include <graph/attr\_value.h\>
-   库文件：libgraph\_base.so

## 功能说明<a name="section3729174918713"></a>

获取属性key-value键值对中的value值，并将value值从T类型转换为DT类型。

-   支持将INT类型转换为int64\_t类型
-   支持将FLOAT类型转换为float类型
-   支持将STR类型转换为std::string类型

## 函数原型<a name="section84161445741"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
template<typename T, typename DT>
graphStatus GetValue(DT &val) const
graphStatus GetValue(AscendString &val)
```

## 参数说明<a name="zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="21.68%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22.57%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="55.75%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="21.68%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0182636394_p34957489"><a name="zh-cn_topic_0182636394_p34957489"></a><a name="zh-cn_topic_0182636394_p34957489"></a>val</p>
</td>
<td class="cellrowborder" valign="top" width="22.57%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0182636394_p12984378"><a name="zh-cn_topic_0182636394_p12984378"></a><a name="zh-cn_topic_0182636394_p12984378"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="55.75%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0182636394_p15921183410"></a>DT类型的参数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section413535858"></a>

graphStatus类型：数据类型转换成功，返回GRAPH\_SUCCESS， 否则，返回GRAPH\_FAILED。

## 异常处理<a name="section1548781517515"></a>

无。

## 约束说明<a name="section2021419196520"></a>

无。

