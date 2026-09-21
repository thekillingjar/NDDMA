# INFER\_FORMAT\_FUNC\_REG<a name="ZH-CN_TOPIC_0000002484471494"></a>

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

## 功能说明<a name="zh-cn_topic_0000001264924934_section4383141161113"></a>

注册算子的InferFormat实现。

GE会在整图的Shape与Dtype推导前后分别调用一次整图的InferFormat，过程中会分别调用各个算子的InferFormat函数。如果算子没有注册InferFormat函数，GE将使用默认的推导函数，即输出的Format等于输入的Format。

## 函数原型<a name="zh-cn_topic_0000001264924934_section1762951313516"></a>

```
#define INFER_FORMAT_FUNC_REG(op_name, x) \
__INFER_FORMAT_FUNC_REG_IMPL__(op_name, INFER_FORMAT_FUNC(op_name, x), __COUNTER__)
```

## 参数说明<a name="zh-cn_topic_0000001264924934_section9220201621817"></a>

<a name="zh-cn_topic_0000001264924934_table18350576"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001264924934_row29692563"><th class="cellrowborder" valign="top" width="25.11%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001264924934_p56287378"><a name="zh-cn_topic_0000001264924934_p56287378"></a><a name="zh-cn_topic_0000001264924934_p56287378"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="16.939999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001264924934_p62983758"><a name="zh-cn_topic_0000001264924934_p62983758"></a><a name="zh-cn_topic_0000001264924934_p62983758"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.95%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001264924934_p47160602"><a name="zh-cn_topic_0000001264924934_p47160602"></a><a name="zh-cn_topic_0000001264924934_p47160602"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001264924934_row61912443"><td class="cellrowborder" valign="top" width="25.11%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264924934_p9757144353"><a name="zh-cn_topic_0000001264924934_p9757144353"></a><a name="zh-cn_topic_0000001264924934_p9757144353"></a>op_name</p>
</td>
<td class="cellrowborder" valign="top" width="16.939999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264924934_p13947197203216"><a name="zh-cn_topic_0000001264924934_p13947197203216"></a><a name="zh-cn_topic_0000001264924934_p13947197203216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.95%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264924934_p1476526254"><a name="zh-cn_topic_0000001264924934_p1476526254"></a><a name="zh-cn_topic_0000001264924934_p1476526254"></a>算子类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001264924934_row168464453311"><td class="cellrowborder" valign="top" width="25.11%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001264924934_p147558420513"><a name="zh-cn_topic_0000001264924934_p147558420513"></a><a name="zh-cn_topic_0000001264924934_p147558420513"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="16.939999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001264924934_p209461672321"><a name="zh-cn_topic_0000001264924934_p209461672321"></a><a name="zh-cn_topic_0000001264924934_p209461672321"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.95%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001264924934_p47631661757"><a name="zh-cn_topic_0000001264924934_p47631661757"></a><a name="zh-cn_topic_0000001264924934_p47631661757"></a>InferFormat函数名，使用IMPLEMT_INFERFORMAT_FUNC中的func_name。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001264924934_section9220616191817"></a>

无

## 约束说明<a name="zh-cn_topic_0000001264924934_section192201016191813"></a>

无

## 调用示例和相关API<a name="zh-cn_topic_0000001264924934_section1653173494814"></a>

```
INFER_FORMAT_FUNC_REG(Transpose, TransposeInferFormat);
```

