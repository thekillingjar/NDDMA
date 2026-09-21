# EsCreateGraphInputWithDetails<a name="ZH-CN_TOPIC_0000002486895306"></a>

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

-   头文件：\#include <ge/es\_funcs.h\>
-   库文件：libeager\_style\_graph\_builder\_base.so、libeager\_style\_graph\_builder\_base.a

## 功能说明<a name="section44282627"></a>

创建一个图上的输入。

## 函数原型<a name="section1831611148519"></a>

```
EsCTensorHolder *EsCreateGraphInputWithDetails(EsCGraphBuilder *graph, int64_t index, const char *name, const char *type, C_DataType data_type, C_Format format, const int64_t *shape, int64_t dim_num);
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="10.73%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.47%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p18388115114282"><a name="p18388115114282"></a><a name="p18388115114282"></a>graph</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p165822162612"><a name="p165822162612"></a><a name="p165822162612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p1893111633018"><a name="p1893111633018"></a><a name="p1893111633018"></a>图构建器指针。</p>
</td>
</tr>
<tr id="row107217586567"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p133071359172818"><a name="p133071359172818"></a><a name="p133071359172818"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p9868144912292"><a name="p9868144912292"></a><a name="p9868144912292"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p11581191613619"><a name="p11581191613619"></a><a name="p11581191613619"></a>第几个图的输入，从0开始计数。</p>
</td>
</tr>
<tr id="row821633613614"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p1911412422910"><a name="p1911412422910"></a><a name="p1911412422910"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p1467395032918"><a name="p1467395032918"></a><a name="p1467395032918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p655715411315"><a name="p655715411315"></a><a name="p655715411315"></a>输入的名字，如果为空则默认为input_{index}。</p>
</td>
</tr>
<tr id="row1191511616299"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p1575110123295"><a name="p1575110123295"></a><a name="p1575110123295"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p3369151192910"><a name="p3369151192910"></a><a name="p3369151192910"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p167781685315"><a name="p167781685315"></a><a name="p167781685315"></a>输入的类型，如果为空则默认为Data。</p>
</td>
</tr>
<tr id="row19667191615294"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p165011723122920"><a name="p165011723122920"></a><a name="p165011723122920"></a>data_type</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p797755112918"><a name="p797755112918"></a><a name="p797755112918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p11350171315317"><a name="p11350171315317"></a><a name="p11350171315317"></a>数据类型。</p>
</td>
</tr>
<tr id="row458819268292"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p16038321296"><a name="p16038321296"></a><a name="p16038321296"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p13591115218294"><a name="p13591115218294"></a><a name="p13591115218294"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p4287141773115"><a name="p4287141773115"></a><a name="p4287141773115"></a>格式类型。</p>
</td>
</tr>
<tr id="row1623416353293"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p162704272912"><a name="p162704272912"></a><a name="p162704272912"></a>shape</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p93485536291"><a name="p93485536291"></a><a name="p93485536291"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p13689122214313"><a name="p13689122214313"></a><a name="p13689122214313"></a>shape信息，shape如果为空则为标量。</p>
</td>
</tr>
<tr id="row766812410588"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p76685455816"><a name="p76685455816"></a><a name="p76685455816"></a>dim_num</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p96681244580"><a name="p96681244580"></a><a name="p96681244580"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p196682413588"><a name="p196682413588"></a><a name="p196682413588"></a>维度数量。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.06%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.55999999999999%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.1.4.1.2 "><p id="p177602693214"><a name="p177602693214"></a><a name="p177602693214"></a>EsCTensorHolder *</p>
</td>
<td class="cellrowborder" valign="top" width="76.55999999999999%" headers="mcps1.1.4.1.3 "><p id="p158628310321"><a name="p158628310321"></a><a name="p158628310321"></a>返回的Tensor资源被图构建器管理。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

## 调用示例<a name="section16305113853313"></a>

```
#include "c_types.h" //包含c的枚举类型定义
// 1. 创建图构建器（EsCGraphBuilder）
EsCGraphBuilder *builder = EsCreateGraphBuilder("graph_name");
// 2. 准备Shape(维度信息)
// 例如：Batch=1, Channel=3, Height=224, Width=224
int64_t input_shape[] = {1, 3, 224, 224};

// 计算维度数量(dim_num)
int64_t dim_num = sizeof(input_shape) / sizeof(input_shape[0]); // 结果为 4

// 3. 准备其他参数
int64_t input_index = 0;           // 第 0 个输入
const char *input_name = "data";   // 输入节点名称
const char *node_type = "Data";   // 输入的节点类型
C_DataType dtype = C_DT_FLOAT;   // 数据类型为 float
C_Format fmt = C_FORMAT_NCHW;         // 数据排布为 NCHW

// 4. 调用函数
EsCTensorHolder *input_tensor = EsCreateGraphInputWithDetails(
    builder,      // graph: 图构建器指针
    input_index,  // index: 输入索引
    input_name,   // name: 输入名称
    node_type,    // type: 节点类型字符串
    dtype,        // data_type: 数据类型枚举
    fmt,          // format: 格式枚举
    input_shape,  // shape: 维度数组指针
    dim_num       // dim_num: 维度数组的长度
);
```

