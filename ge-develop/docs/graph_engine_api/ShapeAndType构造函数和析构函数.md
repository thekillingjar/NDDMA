# ShapeAndType构造函数和析构函数<a name="ZH-CN_TOPIC_0000002531286577"></a>

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

## 头文件<a name="section1194194162918"></a>

\#include <graph/inference\_context.h\>

## 函数功能<a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section36893359"></a>

ShapeAndType构造函数和析构函数。

## 函数原型<a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section136951948195410"></a>

```
ShapeAndType()
~ShapeAndType()
ShapeAndType(const Shape &shape, DataType data_type)
```

## 参数说明<a name="zh-cn_topic_0000001265403638_section144401754174018"></a>

<a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="29.59%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="42.78%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p34957489"></a>shape</p>
</td>
<td class="cellrowborder" valign="top" width="29.59%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p12984378"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="42.78%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a>需设置的shape。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265403638_row27421841194118"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265403638_p33846271325"><a name="zh-cn_topic_0000001265403638_p33846271325"></a><a name="zh-cn_topic_0000001265403638_p33846271325"></a>data_type</p>
</td>
<td class="cellrowborder" valign="top" width="29.59%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265403638_p17209562"><a name="zh-cn_topic_0000001265403638_p17209562"></a><a name="zh-cn_topic_0000001265403638_p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="42.78%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265403638_p137431541144111"><a name="zh-cn_topic_0000001265403638_p137431541144111"></a><a name="zh-cn_topic_0000001265403638_p137431541144111"></a>需设置的DataType。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section35572112"></a>

ShapeAndType构造函数返回ShapeAndType类型的对象。

## 异常处理<a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section51713556"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001265403638_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section62768825"></a>

无。

