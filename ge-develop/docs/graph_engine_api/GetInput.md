# GetInput<a name="ZH-CN_TOPIC_0000002516356787"></a>

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

-   头文件：\#include <ge/fusion/subgraph\_boundary.h\>
-   库文件：libge\_compiler

## 功能说明<a name="section94189264416"></a>

根据索引获取一个子图输入。

## 函数原型<a name="section154184274416"></a>

```
Status GetInput(int64_t index, SubgraphInput &subgraph_input) const
```

## 参数说明<a name="section941822204410"></a>

<a name="table34184210449"></a>
<table><thead align="left"><tr id="row104187244419"><th class="cellrowborder" valign="top" width="13.389999999999999%" id="mcps1.1.4.1.1"><p id="p041818211443"><a name="p041818211443"></a><a name="p041818211443"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.01%" id="mcps1.1.4.1.2"><p id="p124181329443"><a name="p124181329443"></a><a name="p124181329443"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="76.6%" id="mcps1.1.4.1.3"><p id="p1541814254414"><a name="p1541814254414"></a><a name="p1541814254414"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1341815274416"><td class="cellrowborder" valign="top" width="13.389999999999999%" headers="mcps1.1.4.1.1 "><p id="p443611111822"><a name="p443611111822"></a><a name="p443611111822"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="10.01%" headers="mcps1.1.4.1.2 "><p id="p1841811220447"><a name="p1841811220447"></a><a name="p1841811220447"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.6%" headers="mcps1.1.4.1.3 "><p id="p14409817102212"><a name="p14409817102212"></a><a name="p14409817102212"></a>输入索引。</p>
</td>
</tr>
<tr id="row16639132018408"><td class="cellrowborder" valign="top" width="13.389999999999999%" headers="mcps1.1.4.1.1 "><p id="p156404206402"><a name="p156404206402"></a><a name="p156404206402"></a>subgraph_input</p>
</td>
<td class="cellrowborder" valign="top" width="10.01%" headers="mcps1.1.4.1.2 "><p id="p1640220204018"><a name="p1640220204018"></a><a name="p1640220204018"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="76.6%" headers="mcps1.1.4.1.3 "><p id="p11411191712223"><a name="p11411191712223"></a><a name="p11411191712223"></a>子图输入Tensor的描述。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section54187204410"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="12.629999999999999%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="75.57000000000001%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="12.629999999999999%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p735191710216"><a name="p735191710216"></a><a name="p735191710216"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="75.57000000000001%" headers="mcps1.1.4.1.3 "><p id="p1619336535"><a name="p1619336535"></a><a name="p1619336535"></a>SUCCESS：获取成功</p>
<p id="p81282695814"><a name="p81282695814"></a><a name="p81282695814"></a>FAILED：获取失败</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section44181722442"></a>

无

