# SetShape<a name="ZH-CN_TOPIC_0000002531206549"></a>

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

## 头文件<a name="section1194194162918"></a>

\#include <graph/inference\_context.h\>

## 函数功能<a name="zh-cn_topic_0000001265084806_section44282627"></a>

设置ShapeAndType类的shape。

## 函数原型<a name="zh-cn_topic_0000001265084806_section1831611148519"></a>

```
void SetShape(const Shape &shape)
```

## 参数说明<a name="zh-cn_topic_0000001265084806_section62999330"></a>

<a name="zh-cn_topic_0000001265084806_table10309404"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265084806_row47530006"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265084806_p24725298"><a name="zh-cn_topic_0000001265084806_p24725298"></a><a name="zh-cn_topic_0000001265084806_p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="26.14%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265084806_p56592155"><a name="zh-cn_topic_0000001265084806_p56592155"></a><a name="zh-cn_topic_0000001265084806_p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.23%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265084806_p54897010"><a name="zh-cn_topic_0000001265084806_p54897010"></a><a name="zh-cn_topic_0000001265084806_p54897010"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265084806_row17472816"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265084806_p6011995"><a name="zh-cn_topic_0000001265084806_p6011995"></a><a name="zh-cn_topic_0000001265084806_p6011995"></a>shape</p>
</td>
<td class="cellrowborder" valign="top" width="26.14%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265084806_p17209562"><a name="zh-cn_topic_0000001265084806_p17209562"></a><a name="zh-cn_topic_0000001265084806_p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.23%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265084806_p34829302"><a name="zh-cn_topic_0000001265084806_p34829302"></a><a name="zh-cn_topic_0000001265084806_p34829302"></a>设置的目标shape。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001265084806_section30123063"></a>

无。

## 异常处理<a name="zh-cn_topic_0000001265084806_section2672115"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001265084806_section24049039"></a>

无。

