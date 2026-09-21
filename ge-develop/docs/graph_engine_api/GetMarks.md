# GetMarks<a name="ZH-CN_TOPIC_0000002499360452"></a>

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

## 功能说明<a name="zh-cn_topic_0000001264924930_section5719164819393"></a>

在资源类算子推理的上下文中，获取成对资源算子的标记。

## 函数原型<a name="zh-cn_topic_0000001264924930_section129319456392"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
const std::vector<std::string> &GetMarks() const
void GetMarks(std::vector<AscendString> &marks) const
```

## 参数说明<a name="zh-cn_topic_0000001264924930_section1618815223918"></a>

<a name="zh-cn_topic_0000001264924930_table97289253814"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001264924930_row167296283810"><th class="cellrowborder" valign="top" width="9.04%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001264924930_p10729327387"><a name="zh-cn_topic_0000001264924930_p10729327387"></a><a name="zh-cn_topic_0000001264924930_p10729327387"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001264924930_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001264924930_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001264924930_zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="66.74%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001264924930_p07296215380"><a name="zh-cn_topic_0000001264924930_p07296215380"></a><a name="zh-cn_topic_0000001264924930_p07296215380"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001264924930_row57292233816"><td class="cellrowborder" valign="top" width="9.04%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264924930_p149111015380"><a name="zh-cn_topic_0000001264924930_p149111015380"></a><a name="zh-cn_topic_0000001264924930_p149111015380"></a>marks</p>
</td>
<td class="cellrowborder" valign="top" width="24.22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264924930_p910493563816"><a name="zh-cn_topic_0000001264924930_p910493563816"></a><a name="zh-cn_topic_0000001264924930_p910493563816"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="66.74%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264924930_p57290263820"><a name="zh-cn_topic_0000001264924930_p57290263820"></a><a name="zh-cn_topic_0000001264924930_p57290263820"></a>资源类算子的标记。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001264924930_section12573195503920"></a>

资源类算子的标记。

## 异常处理<a name="zh-cn_topic_0000001264924930_section66861205403"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001264924930_section1286618413401"></a>

无。

