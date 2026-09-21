# IMPLEMT\_INFERFORMAT\_FUNC<a name="ZH-CN_TOPIC_0000002484311524"></a>

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

## 头文件<a name="section995734502317"></a>

\#include <graph/operator\_reg.h\>

## 功能说明<a name="zh-cn_topic_0000001265403622_section212607105720"></a>

封装算子的inferFormat函数。

## 函数原型<a name="zh-cn_topic_0000001265403622_section129451113125413"></a>

```
IMPLEMT_INFERFORMAT_FUNC(op_name, func_name)
```

## 参数说明<a name="zh-cn_topic_0000001265403622_section1843519321017"></a>

<a name="zh-cn_topic_0000001265403622_table18350576"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265403622_row29692563"><th class="cellrowborder" valign="top" width="25.3%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265403622_p56287378"><a name="zh-cn_topic_0000001265403622_p56287378"></a><a name="zh-cn_topic_0000001265403622_p56287378"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22.770000000000003%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265403622_p62983758"><a name="zh-cn_topic_0000001265403622_p62983758"></a><a name="zh-cn_topic_0000001265403622_p62983758"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="51.93%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265403622_p47160602"><a name="zh-cn_topic_0000001265403622_p47160602"></a><a name="zh-cn_topic_0000001265403622_p47160602"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265403622_row61912443"><td class="cellrowborder" valign="top" width="25.3%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265403622_p9757144353"><a name="zh-cn_topic_0000001265403622_p9757144353"></a><a name="zh-cn_topic_0000001265403622_p9757144353"></a>op_name</p>
</td>
<td class="cellrowborder" valign="top" width="22.770000000000003%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265403622_p13947197203216"><a name="zh-cn_topic_0000001265403622_p13947197203216"></a><a name="zh-cn_topic_0000001265403622_p13947197203216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.93%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265403622_p1476526254"><a name="zh-cn_topic_0000001265403622_p1476526254"></a><a name="zh-cn_topic_0000001265403622_p1476526254"></a>算子类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265403622_row168464453311"><td class="cellrowborder" valign="top" width="25.3%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265403622_p147558420513"><a name="zh-cn_topic_0000001265403622_p147558420513"></a><a name="zh-cn_topic_0000001265403622_p147558420513"></a>func_name</p>
</td>
<td class="cellrowborder" valign="top" width="22.770000000000003%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265403622_p209461672321"><a name="zh-cn_topic_0000001265403622_p209461672321"></a><a name="zh-cn_topic_0000001265403622_p209461672321"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="51.93%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265403622_p47631661757"><a name="zh-cn_topic_0000001265403622_p47631661757"></a><a name="zh-cn_topic_0000001265403622_p47631661757"></a>inferFormat函数名，用户自定义。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001265403622_section8491941407"></a>

无

## 约束说明<a name="zh-cn_topic_0000001265403622_section65498832"></a>

无

