# SetInputs<a name="ZH-CN_TOPIC_0000002516516737"></a>

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

-   头文件：\#include <graph/graph.h\>
-   库文件：libgraph.so

## 功能说明<a name="section15686020"></a>

设置Graph内的输入算子。

## 函数原型<a name="section1610715016501"></a>

```
Graph &SetInputs(const std::vector<Operator> &inputs)
```

## 参数说明<a name="section6956456"></a>

<a name="table33761356"></a>
<table><thead align="left"><tr id="row27598891"><th class="cellrowborder" valign="top" width="10.41%" id="mcps1.1.4.1.1"><p id="p20917673"><a name="p20917673"></a><a name="p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.73%" id="mcps1.1.4.1.2"><p id="p16609919"><a name="p16609919"></a><a name="p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.86%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row27795493"><td class="cellrowborder" valign="top" width="10.41%" headers="mcps1.1.4.1.1 "><p id="p36842478"><a name="p36842478"></a><a name="p36842478"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="13.73%" headers="mcps1.1.4.1.2 "><p id="p31450711"><a name="p31450711"></a><a name="p31450711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.86%" headers="mcps1.1.4.1.3 "><p id="p55467050"><a name="p55467050"></a><a name="p55467050"></a>Graph内的输入算子。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section62608109"></a>

<a name="table63646052"></a>
<table><thead align="left"><tr id="row46496694"><th class="cellrowborder" valign="top" width="11.110000000000001%" id="mcps1.1.4.1.1"><p id="p8135907"><a name="p8135907"></a><a name="p8135907"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.39%" id="mcps1.1.4.1.2"><p id="p55028720"><a name="p55028720"></a><a name="p55028720"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="74.5%" id="mcps1.1.4.1.3"><p id="p28141317"><a name="p28141317"></a><a name="p28141317"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row64854219"><td class="cellrowborder" valign="top" width="11.110000000000001%" headers="mcps1.1.4.1.1 "><p id="p18700393"><a name="p18700393"></a><a name="p18700393"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.39%" headers="mcps1.1.4.1.2 "><p id="p38336904"><a name="p38336904"></a><a name="p38336904"></a><a href="Graph.md">Graph</a>&amp;</p>
</td>
<td class="cellrowborder" valign="top" width="74.5%" headers="mcps1.1.4.1.3 "><p id="p18281552"><a name="p18281552"></a><a name="p18281552"></a>返回调用者本身。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section38092103"></a>

无

