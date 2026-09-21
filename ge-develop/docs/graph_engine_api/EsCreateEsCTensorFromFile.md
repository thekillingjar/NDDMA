# EsCreateEsCTensorFromFile<a name="ZH-CN_TOPIC_0000002488105194"></a>

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

用于C用户通过binary文件创建EsCTensor。

## 函数原型<a name="section1831611148519"></a>

```
EsCTensor *EsCreateEsCTensorFromFile(const char *data_file_path, const int64_t *dim, int64_t dim_num,C_DataType data_type, C_Format format)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="16.25%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.27%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.48%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="16.25%" headers="mcps1.1.4.1.1 "><p id="p1583161612618"><a name="p1583161612618"></a><a name="p1583161612618"></a>data_file_path</p>
</td>
<td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.1.4.1.2 "><p id="p165822162612"><a name="p165822162612"></a><a name="p165822162612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.48%" headers="mcps1.1.4.1.3 "><p id="p185820162619"><a name="p185820162619"></a><a name="p185820162619"></a>张量binary数据文件路径。</p>
</td>
</tr>
<tr id="row245610301195"><td class="cellrowborder" valign="top" width="16.25%" headers="mcps1.1.4.1.1 "><p id="p13457113011916"><a name="p13457113011916"></a><a name="p13457113011916"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.1.4.1.2 "><p id="p144110415203"><a name="p144110415203"></a><a name="p144110415203"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.48%" headers="mcps1.1.4.1.3 "><p id="p1345715305194"><a name="p1345715305194"></a><a name="p1345715305194"></a>张量维度数组指针。</p>
</td>
</tr>
<tr id="row9556113811917"><td class="cellrowborder" valign="top" width="16.25%" headers="mcps1.1.4.1.1 "><p id="p5556738131914"><a name="p5556738131914"></a><a name="p5556738131914"></a>dim_num</p>
</td>
<td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.1.4.1.2 "><p id="p137017510203"><a name="p137017510203"></a><a name="p137017510203"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.48%" headers="mcps1.1.4.1.3 "><p id="p6556138131918"><a name="p6556138131918"></a><a name="p6556138131918"></a>张量维度数量。</p>
</td>
</tr>
<tr id="row915234110192"><td class="cellrowborder" valign="top" width="16.25%" headers="mcps1.1.4.1.1 "><p id="p19152241161918"><a name="p19152241161918"></a><a name="p19152241161918"></a>data_type</p>
</td>
<td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.1.4.1.2 "><p id="p295915511200"><a name="p295915511200"></a><a name="p295915511200"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.48%" headers="mcps1.1.4.1.3 "><p id="p61521441101917"><a name="p61521441101917"></a><a name="p61521441101917"></a>张量的DataType枚举值。</p>
</td>
</tr>
<tr id="row4475134312190"><td class="cellrowborder" valign="top" width="16.25%" headers="mcps1.1.4.1.1 "><p id="p194751943161919"><a name="p194751943161919"></a><a name="p194751943161919"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.1.4.1.2 "><p id="p77101068209"><a name="p77101068209"></a><a name="p77101068209"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.48%" headers="mcps1.1.4.1.3 "><p id="p1247554351918"><a name="p1247554351918"></a><a name="p1247554351918"></a>张量格式。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.06%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.49%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="74.45%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.49%" headers="mcps1.1.4.1.2 "><p id="p1157618141266"><a name="p1157618141266"></a><a name="p1157618141266"></a>EsCTensor</p>
</td>
<td class="cellrowborder" valign="top" width="74.45%" headers="mcps1.1.4.1.3 "><p id="p95757141768"><a name="p95757141768"></a><a name="p95757141768"></a>张量的匿名指针，失败时返回nullptr。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

## 调用示例<a name="section16305113853313"></a>

```
int64_t dims[] = {3};
auto es_tensor =
    EsCreateEsCTensorFromFile(file_path.c_str(), dims, 1, C_DT_INT64, C_FORMAT_ALL);
```

