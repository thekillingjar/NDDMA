# AddGraphClone<a name="ZH-CN_TOPIC_0000002508557555"></a>

## 产品支持情况<a name="section1993214622115"></a>

<a name="zh-cn_topic_0000001265245842_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265245842_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001265245842_p1883113061818"><a name="zh-cn_topic_0000001265245842_p1883113061818"></a><a name="zh-cn_topic_0000001265245842_p1883113061818"></a><span id="zh-cn_topic_0000001265245842_ph20833205312295"><a name="zh-cn_topic_0000001265245842_ph20833205312295"></a><a name="zh-cn_topic_0000001265245842_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001265245842_p783113012187"><a name="zh-cn_topic_0000001265245842_p783113012187"></a><a name="zh-cn_topic_0000001265245842_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265245842_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265245842_p48327011813"><a name="zh-cn_topic_0000001265245842_p48327011813"></a><a name="zh-cn_topic_0000001265245842_p48327011813"></a><span id="zh-cn_topic_0000001265245842_ph583230201815"><a name="zh-cn_topic_0000001265245842_ph583230201815"></a><a name="zh-cn_topic_0000001265245842_ph583230201815"></a><term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265245842_p108715341013"><a name="zh-cn_topic_0000001265245842_p108715341013"></a><a name="zh-cn_topic_0000001265245842_p108715341013"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265245842_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265245842_p14832120181815"><a name="zh-cn_topic_0000001265245842_p14832120181815"></a><a name="zh-cn_topic_0000001265245842_p14832120181815"></a><span id="zh-cn_topic_0000001265245842_ph1483216010188"><a name="zh-cn_topic_0000001265245842_ph1483216010188"></a><a name="zh-cn_topic_0000001265245842_ph1483216010188"></a><term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265245842_p19948143911820"><a name="zh-cn_topic_0000001265245842_p19948143911820"></a><a name="zh-cn_topic_0000001265245842_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <ge/ge\_api\_v2.h\>
-   库文件：libge\_runner\_v2.so

## 功能说明<a name="section44282627"></a>

向GeSession中添加Graph，GeSession内会生成唯一的Graph ID。

相比于[AddGraph](AddGraph-1.md)接口，此接口传入Graph对象后，会产生Graph对象的拷贝。GeSession中保存的图是Graph对象的一个备份，后续对该Graph的修改不影响GeSession内原有Graph，同时GeSession内图的任何修改也不会影响Graph对象。

## 函数原型<a name="section1831611148519"></a>

```
Status AddGraphClone(uint32_t graph_id, const Graph &graph)
Status AddGraphClone(uint32_t graph_id, const Graph &graph, const std::map<AscendString, AscendString> &options)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="11.32%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.23%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.45%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="11.32%" headers="mcps1.1.4.1.1 "><p id="p2020419388582"><a name="p2020419388582"></a><a name="p2020419388582"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="11.23%" headers="mcps1.1.4.1.2 "><p id="p17209562"><a name="p17209562"></a><a name="p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.45%" headers="mcps1.1.4.1.3 "><p id="p34829302"><a name="p34829302"></a><a name="p34829302"></a>Graph对应的ID。</p>
</td>
</tr>
<tr id="row107217586567"><td class="cellrowborder" valign="top" width="11.32%" headers="mcps1.1.4.1.1 "><p id="p0748582562"><a name="p0748582562"></a><a name="p0748582562"></a>graph</p>
</td>
<td class="cellrowborder" valign="top" width="11.23%" headers="mcps1.1.4.1.2 "><p id="p574115855612"><a name="p574115855612"></a><a name="p574115855612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.45%" headers="mcps1.1.4.1.3 "><p id="p7751058165617"><a name="p7751058165617"></a><a name="p7751058165617"></a>需要加载到GeSession内的Graph。</p>
</td>
</tr>
<tr id="row821633613614"><td class="cellrowborder" valign="top" width="11.32%" headers="mcps1.1.4.1.1 "><p id="p112162036163620"><a name="p112162036163620"></a><a name="p112162036163620"></a>options</p>
</td>
<td class="cellrowborder" valign="top" width="11.23%" headers="mcps1.1.4.1.2 "><p id="p19216193603617"><a name="p19216193603617"></a><a name="p19216193603617"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.45%" headers="mcps1.1.4.1.3 "><p id="p18652125115618"><a name="p18652125115618"></a><a name="p18652125115618"></a>map表，key为参数类型，value为参数值，均为字符串格式，描述Graph配置信息。</p>
<p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a>一般情况下可不填，与<a href="GEInitializeV2.md">GEInitializeV2</a>传入的全局options保持一致。</p>
<p id="p155019231136"><a name="p155019231136"></a><a name="p155019231136"></a>如需单独配置当前Graph的配置信息时，可以通过此参数配置，支持的配置项范围和GEInitializeV2一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.459999999999999%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.77%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.77000000000001%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.459999999999999%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.77%" headers="mcps1.1.4.1.2 "><p id="p2071459847"><a name="p2071459847"></a><a name="p2071459847"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="76.77000000000001%" headers="mcps1.1.4.1.3 "><p id="p141243178210"><a name="p141243178210"></a><a name="p141243178210"></a>GE_CLI_GE_NOT_INITIALIZED：GE未初始化。</p>
<p id="p385394232916"><a name="p385394232916"></a><a name="p385394232916"></a>SUCCESS：图添加成功。</p>
<p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>FAILED：图添加失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

相同对象的Graph调用此接口注册，会导致不同的Graph ID对应不同的备份，两个不同Graph ID对应的备份不共享。

