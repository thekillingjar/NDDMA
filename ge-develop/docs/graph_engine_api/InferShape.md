# InferShape<a name="ZH-CN_TOPIC_0000002488136516"></a>

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

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <ge/ge\_utils.h\>
-   库文件：libge\_compiler.so

## 功能说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section36893359"></a>

給定输入shape，对传入的graph做全图shape推导。

本接口只做shape推导，不对图做任何其他优化（如常量折叠、死边消除等）。

## 函数原型<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section136951948195410"></a>

```
static Status InferShape(const Graph &graph, const std::vector<Shape> &input_shape)
```

## 参数说明<a name="section144401754174018"></a>

<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="11.16%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.24%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.60000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p124101727115118"><a name="p124101727115118"></a><a name="p124101727115118"></a>graph</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.1.4.1.2 "><p id="p6410172718518"><a name="p6410172718518"></a><a name="p6410172718518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.60000000000001%" headers="mcps1.1.4.1.3 "><p id="p14409162715114"><a name="p14409162715114"></a><a name="p14409162715114"></a>待推导的图。</p>
</td>
</tr>
<tr id="row24617615522"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p194614617524"><a name="p194614617524"></a><a name="p194614617524"></a>input_shape</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.1.4.1.2 "><p id="p32234136524"><a name="p32234136524"></a><a name="p32234136524"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.60000000000001%" headers="mcps1.1.4.1.3 "><p id="p14461136125219"><a name="p14461136125219"></a><a name="p14461136125219"></a>图输入对应的shape。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section35572112"></a>

<a name="table2187114454420"></a>
<table><thead align="left"><tr id="row16187114419441"><th class="cellrowborder" valign="top" width="16.72%" id="mcps1.1.4.1.1"><p id="p818764415446"><a name="p818764415446"></a><a name="p818764415446"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.86%" id="mcps1.1.4.1.2"><p id="p171875444446"><a name="p171875444446"></a><a name="p171875444446"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.1.4.1.3"><p id="p19187644104412"><a name="p19187644104412"></a><a name="p19187644104412"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17188644174420"><td class="cellrowborder" valign="top" width="16.72%" headers="mcps1.1.4.1.1 "><p id="p51882044184415"><a name="p51882044184415"></a><a name="p51882044184415"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.86%" headers="mcps1.1.4.1.2 "><p id="p11188134414447"><a name="p11188134414447"></a><a name="p11188134414447"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>SUCCESS：推导成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：推导失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section62768825"></a>

无

