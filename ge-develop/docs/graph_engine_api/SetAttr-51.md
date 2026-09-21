# SetAttr<a name="ZH-CN_TOPIC_0000002484276816"></a>

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

-   头文件：\#include <graph/graph.h\>
-   库文件：libgraph.so

## 功能说明<a name="section1152363518157"></a>

设置Graph的属性，泛型属性接口，属性的类型为attr\_value。

## 函数原型<a name="section181551229191510"></a>

```
graphStatus SetAttr(const AscendString &name, const AttrValue &attr_value)
```

## 参数说明<a name="section943616381155"></a>

<a name="table31879444448"></a>
<table><thead align="left"><tr id="row31871044154410"><th class="cellrowborder" valign="top" width="16.59%" id="mcps1.1.4.1.1"><p id="p10187644104410"><a name="p10187644104410"></a><a name="p10187644104410"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.23%" id="mcps1.1.4.1.2"><p id="p1218704444413"><a name="p1218704444413"></a><a name="p1218704444413"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.18%" id="mcps1.1.4.1.3"><p id="p1818774434412"><a name="p1818774434412"></a><a name="p1818774434412"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row13187134414442"><td class="cellrowborder" valign="top" width="16.59%" headers="mcps1.1.4.1.1 "><p id="p8889110146"><a name="p8889110146"></a><a name="p8889110146"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="11.23%" headers="mcps1.1.4.1.2 "><p id="p1388131191420"><a name="p1388131191420"></a><a name="p1388131191420"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.18%" headers="mcps1.1.4.1.3 "><p id="p128711111140"><a name="p128711111140"></a><a name="p128711111140"></a>属性名称。</p>
</td>
</tr>
<tr id="row176028318191"><td class="cellrowborder" valign="top" width="16.59%" headers="mcps1.1.4.1.1 "><p id="p1960211317197"><a name="p1960211317197"></a><a name="p1960211317197"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="11.23%" headers="mcps1.1.4.1.2 "><p id="p141331257124517"><a name="p141331257124517"></a><a name="p141331257124517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.18%" headers="mcps1.1.4.1.3 "><p id="p860219312192"><a name="p860219312192"></a><a name="p860219312192"></a>属性值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1588574118151"></a>

<a name="table2187114454420"></a>
<table><thead align="left"><tr id="row16187114419441"><th class="cellrowborder" valign="top" width="16.72%" id="mcps1.1.4.1.1"><p id="p818764415446"><a name="p818764415446"></a><a name="p818764415446"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.86%" id="mcps1.1.4.1.2"><p id="p171875444446"><a name="p171875444446"></a><a name="p171875444446"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.1.4.1.3"><p id="p19187644104412"><a name="p19187644104412"></a><a name="p19187644104412"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17188644174420"><td class="cellrowborder" valign="top" width="16.72%" headers="mcps1.1.4.1.1 "><p id="p51882044184415"><a name="p51882044184415"></a><a name="p51882044184415"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.86%" headers="mcps1.1.4.1.2 "><p id="p73026092813"><a name="p73026092813"></a><a name="p73026092813"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0235751031_p918814449444"><a name="zh-cn_topic_0235751031_p918814449444"></a><a name="zh-cn_topic_0235751031_p918814449444"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="zh-cn_topic_0235751031_p91883446443"><a name="zh-cn_topic_0235751031_p91883446443"></a><a name="zh-cn_topic_0235751031_p91883446443"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section9739148141518"></a>

无

## 调用示例<a name="section176226441158"></a>

```
AttrValue av;
av.SetAttrValue(static_cast<int64_t>(100));
Graph graph("test");
graph.SetAttr("int_attr", av);
```

