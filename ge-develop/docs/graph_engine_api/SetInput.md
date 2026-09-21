# SetInput<a name="ZH-CN_TOPIC_0000002499520450"></a>

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

## 功能说明<a name="section36893359"></a>

设置算子Input，即由哪个算子的输出连到本算子。

有如下几种SetInput方法：

如果指定srcOprt第0个Output为当前算子Input，使用第一个函数原型设置当前算子Input，不需要指定srcOprt的Output名称。

如果指定srcOprt的其它Output为当前算子Input，使用第二个函数原型设置当前算子Input，需要指定srcOprt的Output名称。

如果指定srcOprt的其它Output为当前算子Input，使用第三个函数原型设置当前算子Input，需要指定srcOprt的第index个Output。

## 函数原型<a name="section136951948195410"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
Operator &SetInput(const std::string &dst_name, const Operator &src_oprt)
Operator &SetInput(const char_t *dst_name, const Operator &src_oprt)
Operator &SetInput(const std::string &dst_name, const Operator &src_oprt, const std::string &name)
Operator &SetInput(const char_t *dst_name, const Operator &src_oprt, const char_t *name)
Operator &SetInput(const std::string &dst_name, const Operator &src_oprt, uint32_t index)
Operator &SetInput(const char_t *dst_name, const Operator &src_oprt, uint32_t index)
Operator &SetInput(uint32_t dst_index, const Operator &src_oprt, uint32_t src_index)
Operator &SetInput(const char_t *dst_name, uint32_t dst_index, const Operator &src_oprt, const char_t *name)
Operator &SetInput(const char_t *dst_name, uint32_t dst_index, const Operator &src_oprt)
```

## 参数说明<a name="section63604780"></a>

<a name="table47561922"></a>
<table><thead align="left"><tr id="row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="p13951479"><a name="p13951479"></a><a name="p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="20.97%" id="mcps1.1.4.1.2"><p id="p56327989"><a name="p56327989"></a><a name="p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="51.4%" id="mcps1.1.4.1.3"><p id="p66531170"><a name="p66531170"></a><a name="p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p34957489"><a name="p34957489"></a><a name="p34957489"></a>dst_name</p>
</td>
<td class="cellrowborder" valign="top" width="20.97%" headers="mcps1.1.4.1.2 "><p id="p12984378"><a name="p12984378"></a><a name="p12984378"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.4%" headers="mcps1.1.4.1.3 "><p id="p15921183410"><a name="p15921183410"></a><a name="p15921183410"></a>当前算子Input名称。</p>
</td>
</tr>
<tr id="row62880759"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p60176744"><a name="p60176744"></a><a name="p60176744"></a>src_oprt</p>
</td>
<td class="cellrowborder" valign="top" width="20.97%" headers="mcps1.1.4.1.2 "><p id="p42478134"><a name="p42478134"></a><a name="p42478134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.4%" headers="mcps1.1.4.1.3 "><p id="p487618394584"><a name="p487618394584"></a><a name="p487618394584"></a>Input名称为dst_name的输入算子对象。</p>
</td>
</tr>
<tr id="row9380549204312"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p738018498434"><a name="p738018498434"></a><a name="p738018498434"></a>src_index</p>
</td>
<td class="cellrowborder" valign="top" width="20.97%" headers="mcps1.1.4.1.2 "><p id="p203801449184311"><a name="p203801449184311"></a><a name="p203801449184311"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.4%" headers="mcps1.1.4.1.3 "><p id="p838094964315"><a name="p838094964315"></a><a name="p838094964315"></a>src_oprt的第src_index个输出。</p>
</td>
</tr>
<tr id="row206301133449"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p37041500"><a name="p37041500"></a><a name="p37041500"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="20.97%" headers="mcps1.1.4.1.2 "><p id="p47571555"><a name="p47571555"></a><a name="p47571555"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.4%" headers="mcps1.1.4.1.3 "><p id="p60754972"><a name="p60754972"></a><a name="p60754972"></a>src_oprt的Output名称。</p>
</td>
</tr>
<tr id="row389111718272"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p118921217172710"><a name="p118921217172710"></a><a name="p118921217172710"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="20.97%" headers="mcps1.1.4.1.2 "><p id="p7892161714275"><a name="p7892161714275"></a><a name="p7892161714275"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.4%" headers="mcps1.1.4.1.3 "><p id="p148925173272"><a name="p148925173272"></a><a name="p148925173272"></a>src_oprt的第index个Output。</p>
</td>
</tr>
<tr id="row148821716106"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p348821719104"><a name="p348821719104"></a><a name="p348821719104"></a>dst_index</p>
</td>
<td class="cellrowborder" valign="top" width="20.97%" headers="mcps1.1.4.1.2 "><p id="p1148851771013"><a name="p1148851771013"></a><a name="p1148851771013"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.4%" headers="mcps1.1.4.1.3 "><p id="p1348818177105"><a name="p1348818177105"></a><a name="p1348818177105"></a>名称为dst_name的第dst_index个动态输入。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section35572112"></a>

当前调度者本身。

## 异常处理<a name="section51713556"></a>

无。

## 约束说明<a name="section62768825"></a>

无。

