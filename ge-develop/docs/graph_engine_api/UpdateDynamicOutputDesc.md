# UpdateDynamicOutputDesc<a name="ZH-CN_TOPIC_0000002499360482"></a>

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

## 功能说明<a name="section57773428"></a>

根据name和index的组合更新算子动态Output的TensorDesc。

## 函数原型<a name="section121417208912"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus UpdateDynamicOutputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc)
graphStatus UpdateDynamicOutputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc)
```

## 参数说明<a name="section50198807"></a>

<a name="table53866394"></a>
<table><thead align="left"><tr id="row29963612"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="p11133493"><a name="p11133493"></a><a name="p11133493"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="23.77%" id="mcps1.1.4.1.2"><p id="p29397739"><a name="p29397739"></a><a name="p29397739"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="48.6%" id="mcps1.1.4.1.3"><p id="p7691275"><a name="p7691275"></a><a name="p7691275"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19013575"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p63704574"><a name="p63704574"></a><a name="p63704574"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="23.77%" headers="mcps1.1.4.1.2 "><p id="p59796890"><a name="p59796890"></a><a name="p59796890"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.6%" headers="mcps1.1.4.1.3 "><p id="p13471132114116"><a name="p13471132114116"></a><a name="p13471132114116"></a>算子动态Output的名称。</p>
</td>
</tr>
<tr id="row13731878"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p38540333"><a name="p38540333"></a><a name="p38540333"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="23.77%" headers="mcps1.1.4.1.2 "><p id="p34759260"><a name="p34759260"></a><a name="p34759260"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.6%" headers="mcps1.1.4.1.3 "><p id="p2567181019414"><a name="p2567181019414"></a><a name="p2567181019414"></a>算子动态Output编号。</p>
</td>
</tr>
<tr id="row42094334"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p54197854"><a name="p54197854"></a><a name="p54197854"></a>tensor_desc</p>
</td>
<td class="cellrowborder" valign="top" width="23.77%" headers="mcps1.1.4.1.2 "><p id="p27950026"><a name="p27950026"></a><a name="p27950026"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.6%" headers="mcps1.1.4.1.3 "><p id="p38709481"><a name="p38709481"></a><a name="p38709481"></a>TensorDesc对象。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section49136085"></a>

graphStatus类型：更新动态Output成功，返回GRAPH\_SUCCESS，否则，返回GRAPH\_FAILED。

## 异常处理<a name="section39571581"></a>

无。

## 约束说明<a name="section20599914"></a>

无。

