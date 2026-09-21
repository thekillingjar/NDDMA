# aclgrphConvertToWeightRefreshableGraphs<a name="ZH-CN_TOPIC_0000001969686672"></a>

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

-   头文件：\#include <ge/ge\_ir\_build.h\>
-   库文件：libge\_compiler.so

## 功能说明<a name="section44282627"></a>

通过传入Const节点名数组将原图转换成一组权重可更新的图。

该接口适用于权重更新场景。

## 函数原型<a name="section1831611148519"></a>

```
graphStatus aclgrphConvertToWeightRefreshableGraphs(const ge::Graph &origin_graph, const std::vector<AscendString> &const_names, WeightRefreshableGraphs &weight_refreshable_graphs)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="19.21%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.99%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.8%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="19.21%" headers="mcps1.1.4.1.1 "><p id="p10119654151218"><a name="p10119654151218"></a><a name="p10119654151218"></a>origin_graph</p>
</td>
<td class="cellrowborder" valign="top" width="8.99%" headers="mcps1.1.4.1.2 "><p id="p4119165411127"><a name="p4119165411127"></a><a name="p4119165411127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.8%" headers="mcps1.1.4.1.3 "><p id="p2011812542123"><a name="p2011812542123"></a><a name="p2011812542123"></a>需要权重更新的原图。</p>
</td>
</tr>
<tr id="row12682184214272"><td class="cellrowborder" valign="top" width="19.21%" headers="mcps1.1.4.1.1 "><p id="p768234222720"><a name="p768234222720"></a><a name="p768234222720"></a>const_names</p>
</td>
<td class="cellrowborder" valign="top" width="8.99%" headers="mcps1.1.4.1.2 "><p id="p106821342162710"><a name="p106821342162710"></a><a name="p106821342162710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.8%" headers="mcps1.1.4.1.3 "><p id="p86831142132710"><a name="p86831142132710"></a><a name="p86831142132710"></a>需要权重更新的节点名数组。</p>
</td>
</tr>
<tr id="row218824610272"><td class="cellrowborder" valign="top" width="19.21%" headers="mcps1.1.4.1.1 "><p id="p17188134611276"><a name="p17188134611276"></a><a name="p17188134611276"></a>weight_refreshable_graphs</p>
</td>
<td class="cellrowborder" valign="top" width="8.99%" headers="mcps1.1.4.1.2 "><p id="p2018817463278"><a name="p2018817463278"></a><a name="p2018817463278"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.8%" headers="mcps1.1.4.1.3 "><p id="p15188124618275"><a name="p15188124618275"></a><a name="p15188124618275"></a>权重可更新的图，该参数为一结构体，包括三部分：权重初始化图，权重更新图，推理图。</p>
<a name="screen18163730192614"></a><a name="screen18163730192614"></a><pre class="screen" codetype="Cpp" id="screen18163730192614">struct WeightRefreshableGraphs {
  ge::Graph infer_graph;
  ge::Graph var_init_graph;
  ge::Graph var_update_graph;
};</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.78%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.089999999999998%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="75.13%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.78%" headers="mcps1.1.4.1.1 "><p id="p162892055131218"><a name="p162892055131218"></a><a name="p162892055131218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.089999999999998%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>graphStatus<strong id="b3686432181711"><a name="b3686432181711"></a><a name="b3686432181711"></a></strong></p>
</td>
<td class="cellrowborder" valign="top" width="75.13%" headers="mcps1.1.4.1.3 "><p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 调用示例<a name="section1042752318432"></a>

完整调用示例请参见[调用示例](aclgrphBundleSaveModel.md#section117112710367)。

