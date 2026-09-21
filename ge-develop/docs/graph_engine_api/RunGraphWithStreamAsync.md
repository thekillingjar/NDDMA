# RunGraphWithStreamAsync<a name="ZH-CN_TOPIC_0000001264926034"></a>

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

异步运行指定ID对应的Graph图，输出运行结果。该**接口包括了编译，加载和运行Graph的操作**。

此函数与[RunGraph](RunGraph.md)均为运行指定ID对应的图，并输出结果，区别于RunGraph的是，该接口：

-   异步运行。
-   inputs和outputs均为Device上的内存空间，且需要在运行前由用户分配内存大小。

    如下两种情况用户可以不分配输出内存：

    -   用户通过[RegisterExternalAllocator](RegisterExternalAllocator.md)设置了外置allocator，如果没有分配输出内存，由GE调用外置allocator的接口分配内存，用户需要在外置allocator析构前释放这块内存。
    -   用户没有设置外置allocator，动态图场景，如果没有分配输出内存，GE使用内置allocator分配内存，内存的生命周期与图的生命周期保持一致，用户需要在图卸载前（Session析构前、GEFinalize前）主动释放此内存。

## 函数原型<a name="section1831611148519"></a>

```
Status RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<Tensor> &inputs,std::vector<Tensor> &outputs)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="13.07%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.12%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.81%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="13.07%" headers="mcps1.1.4.1.1 "><p id="p6011995"><a name="p6011995"></a><a name="p6011995"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="11.12%" headers="mcps1.1.4.1.2 "><p id="p17209562"><a name="p17209562"></a><a name="p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.81%" headers="mcps1.1.4.1.3 "><p id="p270712592519"><a name="p270712592519"></a><a name="p270712592519"></a>子图对应的ID。</p>
</td>
</tr>
<tr id="row17126854103114"><td class="cellrowborder" valign="top" width="13.07%" headers="mcps1.1.4.1.1 "><p id="p81271554123112"><a name="p81271554123112"></a><a name="p81271554123112"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="11.12%" headers="mcps1.1.4.1.2 "><p id="p5127105419314"><a name="p5127105419314"></a><a name="p5127105419314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.81%" headers="mcps1.1.4.1.3 "><p id="p212712549316"><a name="p212712549316"></a><a name="p212712549316"></a>指定图在哪个Stream上运行。</p>
</td>
</tr>
<tr id="row0533152292412"><td class="cellrowborder" valign="top" width="13.07%" headers="mcps1.1.4.1.1 "><p id="p153312242411"><a name="p153312242411"></a><a name="p153312242411"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="11.12%" headers="mcps1.1.4.1.2 "><p id="p05331222182416"><a name="p05331222182416"></a><a name="p05331222182416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.81%" headers="mcps1.1.4.1.3 "><p id="p1088102316362"><a name="p1088102316362"></a><a name="p1088102316362"></a>当前子图对应的输入数据，为Device上的内存空间。</p>
<p id="p2027391843810"><a name="p2027391843810"></a><a name="p2027391843810"></a>如果通过<span>options</span>指定了ge.exec.hostInputIndexes参数，对应索引的Tensor可以为Host上的内存空间。</p>
</td>
</tr>
<tr id="row7751173518241"><td class="cellrowborder" valign="top" width="13.07%" headers="mcps1.1.4.1.1 "><p id="p275193515244"><a name="p275193515244"></a><a name="p275193515244"></a>outputs</p>
</td>
<td class="cellrowborder" valign="top" width="11.12%" headers="mcps1.1.4.1.2 "><p id="p1475110353246"><a name="p1475110353246"></a><a name="p1475110353246"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.81%" headers="mcps1.1.4.1.3 "><p id="p1764645915393"><a name="p1764645915393"></a><a name="p1764645915393"></a>当前子图对应的输出数据，为Device上的内存空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="13.239999999999998%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.04%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="75.72%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="13.239999999999998%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="75.72%" headers="mcps1.1.4.1.3 "><p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>SUCCESS：异步运行图成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：异步运行图失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

-   调用该接口前，请先分配好Tensor需要使用的内存。
-   调用该接口前，需要通过acl提供的**aclrtCreateStream**接口创建Stream，且只支持Stream为默认Context的场景。
-   得到输出运行结果前，需要通过**aclrtSynchronizeStream**接口保证Stream上的任务已经执行完。

    接口详细说明请参见《应用开发指南 \(C&C++\)》。

