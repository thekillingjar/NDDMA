# GEStreamAllocationSummaryGetLogicalStreamIds<a name="ZH-CN_TOPIC_0000002516867395"></a>

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

## 头文件/库文件<a name="section3132657143710"></a>

-   头文件：\#include <ge/ge\_graph\_compile\_summary.h\>
-   库文件：libge\_compiler.so

## 功能说明<a name="zh-cn_topic_0000001586355282_zh-cn_topic_0000001340364674_section36583473819"></a>

获取根图和子图的逻辑流ID。

## 函数原型<a name="zh-cn_topic_0000001586355282_zh-cn_topic_0000001340364674_section13230182415108"></a>

```
ge::Status GEStreamAllocationSummaryGetLogicalStreamIds(const ge::CompiledGraphSummary &compiled_graph_summary, std::map<AscendString, std::vector<int64_t>> &graph_to_logical_stream_ids)
```

## 参数说明<a name="zh-cn_topic_0000001586355282_zh-cn_topic_0000001340364674_section75395119104"></a>

<a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="23.06%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="9.790000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.15%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="23.06%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p8563195616313"></a><span>compiled_graph_summary</span></p>
</td>
<td class="cellrowborder" valign="top" width="9.790000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.15%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p2684123934216"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p2684123934216"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p2684123934216"></a>图编译后的概要信息。</p>
</td>
</tr>
<tr id="row127391719608"><td class="cellrowborder" valign="top" width="23.06%" headers="mcps1.1.4.1.1 "><p id="p97391819606"><a name="p97391819606"></a><a name="p97391819606"></a><span>graph_to_logical_stream_ids</span></p>
</td>
<td class="cellrowborder" valign="top" width="9.790000000000001%" headers="mcps1.1.4.1.2 "><p id="p15739719702"><a name="p15739719702"></a><a name="p15739719702"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.15%" headers="mcps1.1.4.1.3 "><p id="p873910195014"><a name="p873910195014"></a><a name="p873910195014"></a>map格式，key为图名称，value为逻辑流ID的向量，其中索引为逻辑流ID。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001586355282_zh-cn_topic_0000001340364674_section25791320141317"></a>

<a name="table4209123"></a>
<table><thead align="left"><tr id="row28073419"><th class="cellrowborder" valign="top" width="23.119999999999997%" id="mcps1.1.4.1.1"><p id="p59354473"><a name="p59354473"></a><a name="p59354473"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.75%" id="mcps1.1.4.1.2"><p id="p42983048"><a name="p42983048"></a><a name="p42983048"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="66.13%" id="mcps1.1.4.1.3"><p id="p59074896"><a name="p59074896"></a><a name="p59074896"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row20337256"><td class="cellrowborder" valign="top" width="23.119999999999997%" headers="mcps1.1.4.1.1 "><p id="p36705016"><a name="p36705016"></a><a name="p36705016"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="10.75%" headers="mcps1.1.4.1.2 "><p id="p20316329"><a name="p20316329"></a><a name="p20316329"></a><span>ge::Status</span></p>
</td>
<td class="cellrowborder" valign="top" width="66.13%" headers="mcps1.1.4.1.3 "><a name="ul1343942615010"></a><a name="ul1343942615010"></a><ul id="ul1343942615010"><li>SUCCESS：接口调用成功。</li><li>FAILED：接口调用失败</li></ul>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001586355282_zh-cn_topic_0000001340364674_section19165124931511"></a>

无

