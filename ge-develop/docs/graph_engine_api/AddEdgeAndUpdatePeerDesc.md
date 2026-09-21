# AddEdgeAndUpdatePeerDesc<a name="ZH-CN_TOPIC_0000002520145173"></a>

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

-   头文件：\#include <ge/compliant\_node\_builder.h\>
-   库文件：libeager\_style\_graph\_builder\_base.so、libeager\_style\_graph\_builder\_base\_static.a

## 功能说明<a name="section44282627"></a>

添加边并更新对向Tensor的描述信息

## 函数原型<a name="section1831611148519"></a>

```
graphStatus AddEdgeAndUpdatePeerDesc(Graph &graph, GNode &src_node, int32_t src_port_index, GNode &dst_node, int32_t dst_port_index)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="13.66%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.57%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.77000000000001%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="13.66%" headers="mcps1.1.4.1.1 "><p id="p1583161612618"><a name="p1583161612618"></a><a name="p1583161612618"></a>graph</p>
</td>
<td class="cellrowborder" valign="top" width="10.57%" headers="mcps1.1.4.1.2 "><p id="p165822162612"><a name="p165822162612"></a><a name="p165822162612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.77000000000001%" headers="mcps1.1.4.1.3 "><p id="p185820162619"><a name="p185820162619"></a><a name="p185820162619"></a>源节点src_node和目标节点dst_node所属图对象。</p>
</td>
</tr>
<tr id="row37171319153314"><td class="cellrowborder" valign="top" width="13.66%" headers="mcps1.1.4.1.1 "><p id="p971714192335"><a name="p971714192335"></a><a name="p971714192335"></a>src_node</p>
</td>
<td class="cellrowborder" valign="top" width="10.57%" headers="mcps1.1.4.1.2 "><p id="p146627402330"><a name="p146627402330"></a><a name="p146627402330"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.77000000000001%" headers="mcps1.1.4.1.3 "><p id="p19717161916336"><a name="p19717161916336"></a><a name="p19717161916336"></a>源节点。</p>
</td>
</tr>
<tr id="row2400111816333"><td class="cellrowborder" valign="top" width="13.66%" headers="mcps1.1.4.1.1 "><p id="p194001618113320"><a name="p194001618113320"></a><a name="p194001618113320"></a>src_port_index</p>
</td>
<td class="cellrowborder" valign="top" width="10.57%" headers="mcps1.1.4.1.2 "><p id="p9369174143312"><a name="p9369174143312"></a><a name="p9369174143312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.77000000000001%" headers="mcps1.1.4.1.3 "><p id="p195382593419"><a name="p195382593419"></a><a name="p195382593419"></a>源节点端口索引。</p>
</td>
</tr>
<tr id="row893501663311"><td class="cellrowborder" valign="top" width="13.66%" headers="mcps1.1.4.1.1 "><p id="p199351816113312"><a name="p199351816113312"></a><a name="p199351816113312"></a>dst_node</p>
</td>
<td class="cellrowborder" valign="top" width="10.57%" headers="mcps1.1.4.1.2 "><p id="p10789204153315"><a name="p10789204153315"></a><a name="p10789204153315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.77000000000001%" headers="mcps1.1.4.1.3 "><p id="p393515161331"><a name="p393515161331"></a><a name="p393515161331"></a>目标节点。</p>
</td>
</tr>
<tr id="row4925627173310"><td class="cellrowborder" valign="top" width="13.66%" headers="mcps1.1.4.1.1 "><p id="p592512714334"><a name="p592512714334"></a><a name="p592512714334"></a>dst_port_index</p>
</td>
<td class="cellrowborder" valign="top" width="10.57%" headers="mcps1.1.4.1.2 "><p id="p2316144214337"><a name="p2316144214337"></a><a name="p2316144214337"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.77000000000001%" headers="mcps1.1.4.1.3 "><p id="p130113346347"><a name="p130113346347"></a><a name="p130113346347"></a>目标节点端口索引。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.06%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="73.94%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="15%" headers="mcps1.1.4.1.2 "><p id="p1157618141266"><a name="p1157618141266"></a><a name="p1157618141266"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="73.94%" headers="mcps1.1.4.1.3 "><p id="p296018569247"><a name="p296018569247"></a><a name="p296018569247"></a>GRAPH_SUCCESS(0): 成功</p>
<p id="p95757141768"><a name="p95757141768"></a><a name="p95757141768"></a>其他值: 失败</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

