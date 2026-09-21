# GeSessionLoadGraph<a name="ZH-CN_TOPIC_0000002065236701"></a>

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

-   头文件：\#include <ge/ge\_api.h\>
-   库文件：libge\_runner.so

## 功能说明<a name="section44282627"></a>

只有异步执行Graph场景使用，使用指定的Session，将指定Graph ID的图绑定到对应Stream上。

GeSessionLoadGraph成功后可以使用[GeSessionExecuteGraphWithStreamAsync](GeSessionExecuteGraphWithStreamAsync.md)接口执行图。

## 函数原型<a name="section1831611148519"></a>

```
ge::Status GeSessionLoadGraph(ge::Session &session, uint32_t graph_id, const std::map<ge::AscendString, ge::AscendString> &options, void *stream)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="12.78%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.68%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.53999999999999%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1699982643210"><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1 "><p id="p199962615327"><a name="p199962615327"></a><a name="p199962615327"></a>session</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2 "><p id="p16019272324"><a name="p16019272324"></a><a name="p16019272324"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.53999999999999%" headers="mcps1.1.4.1.3 "><p id="p1701277326"><a name="p1701277326"></a><a name="p1701277326"></a>加载图的Session实例。</p>
</td>
</tr>
<tr id="row17472816"><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1 "><p id="p6011995"><a name="p6011995"></a><a name="p6011995"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2 "><p id="p17209562"><a name="p17209562"></a><a name="p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.53999999999999%" headers="mcps1.1.4.1.3 "><p id="p34829302"><a name="p34829302"></a><a name="p34829302"></a>要执行图对应的ID。</p>
</td>
</tr>
<tr id="row0115745088"><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1 "><p id="p611624515811"><a name="p611624515811"></a><a name="p611624515811"></a><span>options</span></p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2 "><p id="p1211615453816"><a name="p1211615453816"></a><a name="p1211615453816"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.53999999999999%" headers="mcps1.1.4.1.3 "><p id="p365171011499"><a name="p365171011499"></a><a name="p365171011499"></a><span>执行阶段可能用到的</span><span>options</span>。map表，key为参数类型，value为参数值，描述Graph配置信息。</p>
<p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a>一般情况下可不填，与GEInitialize传入的全局options保持一致。</p>
<p id="p1231855123210"><a name="p1231855123210"></a><a name="p1231855123210"></a><span>key和value类型为AscendString</span>，如需单独配置当前Graph的配置信息时，可以通过此参数配置，支持的配置项请参见<a href="options参数说明.md">options参数说明</a>&gt;<strong id="b8463133714459"><a name="b8463133714459"></a><a name="b8463133714459"></a>ge.exec.frozenInputIndexes</strong>，当前只支持配置该参数。</p>
</td>
</tr>
<tr id="row188211176016"><td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.4.1.1 "><p id="p48218171605"><a name="p48218171605"></a><a name="p48218171605"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="9.68%" headers="mcps1.1.4.1.2 "><p id="p18821717703"><a name="p18821717703"></a><a name="p18821717703"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.53999999999999%" headers="mcps1.1.4.1.3 "><p id="p13828175014"><a name="p13828175014"></a><a name="p13828175014"></a>图执行流。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="12.139999999999999%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="77.82%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="77.82%" headers="mcps1.1.4.1.3 "><p id="p141243178210"><a name="p141243178210"></a><a name="p141243178210"></a>GE_CLI_SESS_RUN_FAILED：执行子图时序列化转换失败。</p>
<p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>SUCCESS：执行子图成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：执行子图失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

该接口执行前需要完成[CompileGraph](CompileGraph.md)流程，且需要与[GeSessionExecuteGraphWithStreamAsync](GeSessionExecuteGraphWithStreamAsync.md)接口配合使用。

