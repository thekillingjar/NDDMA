# GetAllSubgraphs<a name="ZH-CN_TOPIC_0000002524343461"></a>

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

## 功能说明<a name="section44282627"></a>

获取所有的子图。

## 函数原型<a name="section1831611148519"></a>

```
std::vector<GraphPtr> GetAllSubgraphs() const
```

## 参数说明<a name="section62999330"></a>

无

## 返回值说明<a name="section30123063"></a>

<a name="zh-cn_topic_0235751031_table2601186"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row1832323"><th class="cellrowborder" valign="top" width="11.86%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p14200498"><a name="zh-cn_topic_0235751031_p14200498"></a><a name="zh-cn_topic_0235751031_p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.84%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p9389685"><a name="zh-cn_topic_0235751031_p9389685"></a><a name="zh-cn_topic_0235751031_p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="70.3%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p22367029"><a name="zh-cn_topic_0235751031_p22367029"></a><a name="zh-cn_topic_0235751031_p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0235751031_row66898905"><td class="cellrowborder" valign="top" width="11.86%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0235751031_p50102218"><a name="zh-cn_topic_0235751031_p50102218"></a><a name="zh-cn_topic_0235751031_p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="17.84%" headers="mcps1.1.4.1.2 "><p id="p73026092813"><a name="p73026092813"></a><a name="p73026092813"></a>std::vector&lt;GraphPtr&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.1.4.1.3 "><p id="p1993847122818"><a name="p1993847122818"></a><a name="p1993847122818"></a>获取的所有子图。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

