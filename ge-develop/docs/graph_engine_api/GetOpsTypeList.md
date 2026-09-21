# GetOpsTypeList<a name="ZH-CN_TOPIC_0000002531200381"></a>

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

## 头文件<a name="section18580257113011"></a>

\#include <graph/operator\_factory.h\>

## 功能说明<a name="zh-cn_topic_0000001312484725_section36893359"></a>

获取系统支持的所有算子类型列表。

## 函数原型<a name="zh-cn_topic_0000001312484725_section136951948195410"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
static graphStatus GetOpsTypeList(std::vector<std::string> &all_ops)
static graphStatus GetOpsTypeList(std::vector<AscendString> &all_ops)
```

## 参数说明<a name="zh-cn_topic_0000001312484725_section13189358"></a>

<a name="zh-cn_topic_0000001312484725_table32576140"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312484725_row60665573"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312484725_p14964341"><a name="zh-cn_topic_0000001312484725_p14964341"></a><a name="zh-cn_topic_0000001312484725_p14964341"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="28.12%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312484725_p4152081"><a name="zh-cn_topic_0000001312484725_p4152081"></a><a name="zh-cn_topic_0000001312484725_p4152081"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="44.25%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312484725_p62718864"><a name="zh-cn_topic_0000001312484725_p62718864"></a><a name="zh-cn_topic_0000001312484725_p62718864"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312484725_row47063234"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312484725_p54025633"><a name="zh-cn_topic_0000001312484725_p54025633"></a><a name="zh-cn_topic_0000001312484725_p54025633"></a>all_ops</p>
</td>
<td class="cellrowborder" valign="top" width="28.12%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312484725_p14000148"><a name="zh-cn_topic_0000001312484725_p14000148"></a><a name="zh-cn_topic_0000001312484725_p14000148"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="44.25%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312484725_p50048984"><a name="zh-cn_topic_0000001312484725_p50048984"></a><a name="zh-cn_topic_0000001312484725_p50048984"></a>算子类型列表。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001312484725_section35572112"></a>

graphStatus类型：GRAPH\_SUCCESS，代表成功；GRAPH\_FAILED，代表失败。

## 约束说明<a name="zh-cn_topic_0000001312484725_section62768825"></a>

无。

