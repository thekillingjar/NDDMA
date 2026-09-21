# SetSubgraphBuilder<a name="ZH-CN_TOPIC_0000002499520452"></a>

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

## 头文件和库文件<a name="section094612611340"></a>

-   头文件：\#include <graph/operator.h\>
-   库文件：libgraph.so

## 功能说明<a name="section59140967"></a>

设置指定子图构建的函数对象。

## 函数原型<a name="section15990165911124"></a>

```
void SetSubgraphBuilder(const char_t *ir_name, uint32_t index, const SubgraphBuilder &builder)
```

## 参数说明<a name="section62506663"></a>

<a name="table34416674"></a>
<table><thead align="left"><tr id="row4508766"><th class="cellrowborder" valign="top" width="29.48%" id="mcps1.1.4.1.1"><p id="p29665754"><a name="p29665754"></a><a name="p29665754"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.48%" id="mcps1.1.4.1.2"><p id="p54115834"><a name="p54115834"></a><a name="p54115834"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="61.040000000000006%" id="mcps1.1.4.1.3"><p id="p48097351"><a name="p48097351"></a><a name="p48097351"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row3571397"><td class="cellrowborder" valign="top" width="29.48%" headers="mcps1.1.4.1.1 "><p id="p13958182714280"><a name="p13958182714280"></a><a name="p13958182714280"></a>ir_name</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.1.4.1.2 "><p id="p10946294"><a name="p10946294"></a><a name="p10946294"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.040000000000006%" headers="mcps1.1.4.1.3 "><p id="p147851711143915"><a name="p147851711143915"></a><a name="p147851711143915"></a>子图名称。</p>
</td>
</tr>
<tr id="row119261235152811"><td class="cellrowborder" valign="top" width="29.48%" headers="mcps1.1.4.1.1 "><p id="p7923102963112"><a name="p7923102963112"></a><a name="p7923102963112"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.1.4.1.2 "><p id="p2927143512289"><a name="p2927143512289"></a><a name="p2927143512289"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.040000000000006%" headers="mcps1.1.4.1.3 "><p id="p13361133612315"><a name="p13361133612315"></a><a name="p13361133612315"></a>动态个数子图场景（子图数量不固定），标识子图的序号。</p>
</td>
</tr>
<tr id="row18763740173110"><td class="cellrowborder" valign="top" width="29.48%" headers="mcps1.1.4.1.1 "><p id="p1976454033112"><a name="p1976454033112"></a><a name="p1976454033112"></a>builder</p>
</td>
<td class="cellrowborder" valign="top" width="9.48%" headers="mcps1.1.4.1.2 "><p id="p14764440163111"><a name="p14764440163111"></a><a name="p14764440163111"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.040000000000006%" headers="mcps1.1.4.1.3 "><p id="p276424023114"><a name="p276424023114"></a><a name="p276424023114"></a>子图构建的函数对象。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25689059"></a>

无。

## 异常处理<a name="section29874939"></a>

无。

## 约束说明<a name="section439002"></a>

无。

