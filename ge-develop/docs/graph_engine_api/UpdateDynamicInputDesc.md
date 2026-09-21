# UpdateDynamicInputDesc<a name="ZH-CN_TOPIC_0000002531280425"></a>

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

## 功能说明<a name="section8361080"></a>

根据name和index的组合更新算子动态Input的TensorDesc。

## 函数原型<a name="section18615321583"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus UpdateDynamicInputDesc(const std::string &name, uint32_t index, const TensorDesc &tensor_desc)
graphStatus UpdateDynamicInputDesc(const char_t *name, uint32_t index, const TensorDesc &tensor_desc)
```

## 参数说明<a name="section8140857"></a>

<a name="table17230008"></a>
<table><thead align="left"><tr id="row40248354"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="p38891250"><a name="p38891250"></a><a name="p38891250"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.7%" id="mcps1.1.4.1.2"><p id="p63183526"><a name="p63183526"></a><a name="p63183526"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="47.67%" id="mcps1.1.4.1.3"><p id="p15663836"><a name="p15663836"></a><a name="p15663836"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row60811181"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p26758660"><a name="p26758660"></a><a name="p26758660"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="24.7%" headers="mcps1.1.4.1.2 "><p id="p19967814"><a name="p19967814"></a><a name="p19967814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.67%" headers="mcps1.1.4.1.3 "><p id="p13471132114116"><a name="p13471132114116"></a><a name="p13471132114116"></a>算子动态Input的名称。</p>
</td>
</tr>
<tr id="row43857981"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p62835574"><a name="p62835574"></a><a name="p62835574"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="24.7%" headers="mcps1.1.4.1.2 "><p id="p56516768"><a name="p56516768"></a><a name="p56516768"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.67%" headers="mcps1.1.4.1.3 "><p id="p2567181019414"><a name="p2567181019414"></a><a name="p2567181019414"></a>算子动态Input编号，编号从1开始。</p>
</td>
</tr>
<tr id="row1984791"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p26550365"><a name="p26550365"></a><a name="p26550365"></a>tensor_desc</p>
</td>
<td class="cellrowborder" valign="top" width="24.7%" headers="mcps1.1.4.1.2 "><p id="p3095993"><a name="p3095993"></a><a name="p3095993"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.67%" headers="mcps1.1.4.1.3 "><p id="p45933772"><a name="p45933772"></a><a name="p45933772"></a>TensorDesc对象。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section6158855"></a>

graphStatus类型：更新动态Input成功，返回GRAPH\_SUCCESS， 否则，返回GRAPH\_FAILED。

## 异常处理<a name="section55429698"></a>

无。

## 约束说明<a name="section29105235"></a>

无。

