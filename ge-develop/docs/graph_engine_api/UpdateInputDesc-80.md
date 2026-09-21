# UpdateInputDesc<a name="ZH-CN_TOPIC_0000002499520454"></a>

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

## 功能说明<a name="section35329809"></a>

根据算子Input名称更新Input的TensorDesc。

## 函数原型<a name="section1159353517563"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus UpdateInputDesc(const std::string &name, const TensorDesc &tensor_desc)
graphStatus UpdateInputDesc(const char_t *name, const TensorDesc &tensor_desc)
```

## 参数说明<a name="section49532829"></a>

<a name="table34019058"></a>
<table><thead align="left"><tr id="row16856419"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="p23192694"><a name="p23192694"></a><a name="p23192694"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.03%" id="mcps1.1.4.1.2"><p id="p66668934"><a name="p66668934"></a><a name="p66668934"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="47.339999999999996%" id="mcps1.1.4.1.3"><p id="p66410939"><a name="p66410939"></a><a name="p66410939"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row10576944"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p51426113"><a name="p51426113"></a><a name="p51426113"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="25.03%" headers="mcps1.1.4.1.2 "><p id="p4765656"><a name="p4765656"></a><a name="p4765656"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.339999999999996%" headers="mcps1.1.4.1.3 "><p id="p61847473"><a name="p61847473"></a><a name="p61847473"></a>算子Input名称。</p>
</td>
</tr>
<tr id="row19756352"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p56760689"><a name="p56760689"></a><a name="p56760689"></a>tensor_desc</p>
</td>
<td class="cellrowborder" valign="top" width="25.03%" headers="mcps1.1.4.1.2 "><p id="p34213098"><a name="p34213098"></a><a name="p34213098"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.339999999999996%" headers="mcps1.1.4.1.3 "><p id="p60100823"><a name="p60100823"></a><a name="p60100823"></a>TensorDesc对象。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section43142284"></a>

graphStatus类型：更新TensorDesc成功，返回GRAPH\_SUCCESS， 否则，返回GRAPH\_FAILED。

## 异常处理<a name="section52736237"></a>

<a name="table42184651"></a>
<table><thead align="left"><tr id="row14649520"><th class="cellrowborder" valign="top" width="27.889999999999997%" id="mcps1.1.3.1.1"><p id="p45760497"><a name="p45760497"></a><a name="p45760497"></a>异常场景</p>
</th>
<th class="cellrowborder" valign="top" width="72.11%" id="mcps1.1.3.1.2"><p id="p15612815"><a name="p15612815"></a><a name="p15612815"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row56678513"><td class="cellrowborder" valign="top" width="27.889999999999997%" headers="mcps1.1.3.1.1 "><p id="p27556864"><a name="p27556864"></a><a name="p27556864"></a>无对应name输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.11%" headers="mcps1.1.3.1.2 "><p id="p17513507"><a name="p17513507"></a><a name="p17513507"></a>函数提前结束，返回GRAPH_FAILED。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section4864091"></a>

无。

