# PaRemapped<a name="ZH-CN_TOPIC_0000001936444554"></a>

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

用于内存出现UCE（Uncorrectable Error，不可纠正错误）错误时判断此段内存是否可以快速恢复。

## 函数原型<a name="section1831611148519"></a>

```
Status PaRemapped(const uint64_t va, const uint64_t new_pa, const uint64_t len)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="12.6%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.01%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.39%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.1 "><p id="p6475110121816"><a name="p6475110121816"></a><a name="p6475110121816"></a>va</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.1.4.1.2 "><p id="p5473150131815"><a name="p5473150131815"></a><a name="p5473150131815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.39%" headers="mcps1.1.4.1.3 "><p id="p547017041819"><a name="p547017041819"></a><a name="p547017041819"></a>发生UCE错误的virtual address。</p>
</td>
</tr>
<tr id="row055265114810"><td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.1 "><p id="p175538564820"><a name="p175538564820"></a><a name="p175538564820"></a>new_pa</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.1.4.1.2 "><p id="p10553135184814"><a name="p10553135184814"></a><a name="p10553135184814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.39%" headers="mcps1.1.4.1.3 "><p id="p19553850485"><a name="p19553850485"></a><a name="p19553850485"></a>新申请内存的physical address。预留参数。</p>
</td>
</tr>
<tr id="row364699124818"><td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.1 "><p id="p116467924811"><a name="p116467924811"></a><a name="p116467924811"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.1.4.1.2 "><p id="p1764619184817"><a name="p1764619184817"></a><a name="p1764619184817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.39%" headers="mcps1.1.4.1.3 "><p id="p264616910484"><a name="p264616910484"></a><a name="p264616910484"></a>发生UCE错误的内存大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="12.509999999999998%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.57%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="74.92%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="12.509999999999998%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.57%" headers="mcps1.1.4.1.2 "><p id="p1896616536411"><a name="p1896616536411"></a><a name="p1896616536411"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="74.92%" headers="mcps1.1.4.1.3 "><p id="p541171718495"><a name="p541171718495"></a><a name="p541171718495"></a>SUCCESS：对应的va内存段可快速恢复。</p>
<p id="p84110176499"><a name="p84110176499"></a><a name="p84110176499"></a>FAILED：对应的VA内存段存在不可恢复的地址。</p>
<p id="p0411101724916"><a name="p0411101724916"></a><a name="p0411101724916"></a>PARAM_INVALID：存在未识别到VA地址段。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

-   在调用本接口前，必须先调用[CompileGraph](CompileGraph.md)接口进行图编译。
-   不支持Session中存在Host调度的场景，若Session中存在Host调度图，接口返回失败。
-   不支持静态shape图allocation表中的绝对地址段。
-   该接口只适用于静态编译图，可以通过[GetCompiledGraphSummary](GetCompiledGraphSummary.md)接口中的IsStatic接口获取图是否为静态编译。

