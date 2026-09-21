# SetOutputHandleShapesAndTypes<a name="ZH-CN_TOPIC_0000002499360450"></a>

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

## 功能说明<a name="zh-cn_topic_0000001265403618_section1015915114015"></a>

在推理上下文中，设置算子输出句柄的[ShapeAndType](ShapeAndType.md)。

## 函数原型<a name="zh-cn_topic_0000001265403618_section71592118012"></a>

```
void SetOutputHandleShapesAndTypes(const std::vector<std::vector<ShapeAndType>> &shapes_and_types)
void SetOutputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types)
```

## 参数说明<a name="zh-cn_topic_0000001265403618_section815913120013"></a>

<a name="zh-cn_topic_0000001265403618_table201598112010"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265403618_row8159311901"><th class="cellrowborder" valign="top" width="32.629999999999995%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265403618_p51591818018"><a name="zh-cn_topic_0000001265403618_p51591818018"></a><a name="zh-cn_topic_0000001265403618_p51591818018"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="26.55%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265403618_p1615324594411"><a name="zh-cn_topic_0000001265403618_p1615324594411"></a><a name="zh-cn_topic_0000001265403618_p1615324594411"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="40.82%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265403618_p13159011108"><a name="zh-cn_topic_0000001265403618_p13159011108"></a><a name="zh-cn_topic_0000001265403618_p13159011108"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265403618_row101591411012"><td class="cellrowborder" valign="top" width="32.629999999999995%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265403618_p715912113011"><a name="zh-cn_topic_0000001265403618_p715912113011"></a><a name="zh-cn_topic_0000001265403618_p715912113011"></a>shapes_and_types</p>
</td>
<td class="cellrowborder" valign="top" width="26.55%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265403618_p18154445134414"><a name="zh-cn_topic_0000001265403618_p18154445134414"></a><a name="zh-cn_topic_0000001265403618_p18154445134414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="40.82%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265403618_p58441133175014"><a name="zh-cn_topic_0000001265403618_p58441133175014"></a><a name="zh-cn_topic_0000001265403618_p58441133175014"></a>算子输出句柄的<a href="ShapeAndType.md">ShapeAndType</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001265403618_section111601611804"></a>

无。

## 异常处理<a name="zh-cn_topic_0000001265403618_section916041509"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001265403618_section121601216013"></a>

无。

