# UpdateGraphRefreshableFeatureMemoryBase<a name="ZH-CN_TOPIC_0000002508557571"></a>

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

用于更新除了Fixed之外可刷新的Feature内存基址。

在调用本接口前，必须先调用[CompileGraph](CompileGraph-2.md)接口进行图编译；内存大小从[GetCompiledGraphSummary](GetCompiledGraphSummary-3.md)\>**GetRefreshableFeatureMemorySize**接口中获取。

如果图中存在Fixed内存，且用户没有调用[SetGraphFixedFeatureMemoryBaseWithType](SetGraphFixedFeatureMemoryBaseWithType-14.md)接口设置默认内存类型的Fixed Feature内存基地址，则会自动默认申请Fixed内存。

Feature内存指的是模型执行过程中所需要的中间内存（比如中间节点的输入输出等内存）。

## 函数原型<a name="section1831611148519"></a>

```
Status UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="14.829999999999998%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.320000000000002%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.85000000000001%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.1.4.1.1 "><p id="p12677165265315"><a name="p12677165265315"></a><a name="p12677165265315"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.1.4.1.2 "><p id="p17209562"><a name="p17209562"></a><a name="p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.85000000000001%" headers="mcps1.1.4.1.3 "><p id="p270712592519"><a name="p270712592519"></a><a name="p270712592519"></a>子图对应的ID。</p>
</td>
</tr>
<tr id="row11568144391416"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.1.4.1.1 "><p id="p13568144351410"><a name="p13568144351410"></a><a name="p13568144351410"></a>memory</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.1.4.1.2 "><p id="p11568143191412"><a name="p11568143191412"></a><a name="p11568143191412"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.85000000000001%" headers="mcps1.1.4.1.3 "><p id="p6568843131410"><a name="p6568843131410"></a><a name="p6568843131410"></a>设置的除了Fixed之外可刷新的Feature内存基址。</p>
</td>
</tr>
<tr id="row755964141419"><td class="cellrowborder" valign="top" width="14.829999999999998%" headers="mcps1.1.4.1.1 "><p id="p14560124115144"><a name="p14560124115144"></a><a name="p14560124115144"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="13.320000000000002%" headers="mcps1.1.4.1.2 "><p id="p4560741131418"><a name="p4560741131418"></a><a name="p4560741131418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.85000000000001%" headers="mcps1.1.4.1.3 "><p id="p5146131219328"><a name="p5146131219328"></a><a name="p5146131219328"></a>设置的除了Fixed之外可刷新的Feature内存大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="15.03%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.44%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="71.53%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="15.03%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.44%" headers="mcps1.1.4.1.2 "><p id="p1896616536411"><a name="p1896616536411"></a><a name="p1896616536411"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="71.53%" headers="mcps1.1.4.1.3 "><p id="p141243178210"><a name="p141243178210"></a><a name="p141243178210"></a>SUCCESS：设置成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：设置失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

-   调用CompileGraph完成图编译后，您可以通过GetCompiledGraphSummary接口获取Feature地址是否可刷新，只有Feature地址可刷新的图才支持重复调用本接口刷新Feature地址。
-   该接口只适用于静态编译图，可以通过GetCompiledGraphSummary接口中的**IsStatic**接口获取图是否为静态编译。
-   若使用了本接口，又配置了[RegisterExternalAllocator](RegisterExternalAllocator-8.md)接口，则RegisterExternalAllocator接口不生效。
-   如果通过[SetGraphConstMemoryBase](SetGraphConstMemoryBase-13.md)或SetGraphFixedFeatureMemoryBaseWithType接口将内存类型为MEMORY\_TYPE\_DEFAULT的Fixed内存地址设置为nullptr，size设置为0，表示关闭Fixed内存功能，该场景下无法再调用本接口。
-   不能与[UpdateGraphFeatureMemoryBase](UpdateGraphFeatureMemoryBase-15.md)同时使用。

