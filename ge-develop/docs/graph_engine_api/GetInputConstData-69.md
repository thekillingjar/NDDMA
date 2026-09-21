# GetInputConstData<a name="ZH-CN_TOPIC_0000002499360470"></a>

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

## 头文件和库文件<a name="section094612611340"></a>

-   头文件：\#include <graph/operator.h\>
-   库文件：libgraph.so

## 功能说明<a name="section47701247204112"></a>

如果指定算子Input对应的节点为Const节点，可调用该接口获取Const节点的数据。

## 函数原型<a name="section117235443411"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus GetInputConstData(const std::string &dst_name, Tensor &data) const
graphStatus GetInputConstData(const char_t *dst_name, Tensor &data) const
```

## 参数说明<a name="section149825464113"></a>

<a name="table7988146131516"></a>
<table><thead align="left"><tr id="row139884611519"><th class="cellrowborder" valign="top" width="22.37%" id="mcps1.1.4.1.1"><p id="p99881762158"><a name="p99881762158"></a><a name="p99881762158"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22.689999999999998%" id="mcps1.1.4.1.2"><p id="p698813618150"><a name="p698813618150"></a><a name="p698813618150"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="54.94%" id="mcps1.1.4.1.3"><p id="p18988126121520"><a name="p18988126121520"></a><a name="p18988126121520"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row16988156131510"><td class="cellrowborder" valign="top" width="22.37%" headers="mcps1.1.4.1.1 "><p id="p1798810610155"><a name="p1798810610155"></a><a name="p1798810610155"></a>dst_name</p>
</td>
<td class="cellrowborder" valign="top" width="22.689999999999998%" headers="mcps1.1.4.1.2 "><p id="p298813611519"><a name="p298813611519"></a><a name="p298813611519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="54.94%" headers="mcps1.1.4.1.3 "><p id="p59891164150"><a name="p59891164150"></a><a name="p59891164150"></a>输入名称。</p>
</td>
</tr>
<tr id="row20700318151816"><td class="cellrowborder" valign="top" width="22.37%" headers="mcps1.1.4.1.1 "><p id="p157005189181"><a name="p157005189181"></a><a name="p157005189181"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="22.689999999999998%" headers="mcps1.1.4.1.2 "><p id="p1270012183188"><a name="p1270012183188"></a><a name="p1270012183188"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.94%" headers="mcps1.1.4.1.3 "><p id="p97011818181817"><a name="p97011818181817"></a><a name="p97011818181817"></a>返回Const节点的数据Tensor。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section464035815414"></a>

graphStatus类型：如果指定算子Input对应的节点为Const节点且获取数据成功，返回GRAPH\_SUCCESS，否则，返回GRAPH\_FAILED。

## 异常处理<a name="section11969151184218"></a>

无。

## 约束说明<a name="section9752843422"></a>

无。

