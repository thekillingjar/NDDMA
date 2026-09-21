# BROADCAST\_INFER<a name="ZH-CN_TOPIC_0000002484311520"></a>

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

## 功能说明<a name="zh-cn_topic_0000001265403626_section212607105720"></a>

提供公共函数宏封装，供算子开发者开发InferShape函数。该函数基于2个输入的shape，设置输出的shape。该宏只是设置shape，未设置dtype。

-   如果2个输入的shape一致，会按输入的shape设置输出shape。
-   如果2个输入的shape不一致，会按照Broadcast的策略，取2个输入shape的并集。

    比如输入shape分别为（1,2,3,4）和（3,1,3,4），则该宏会设置算子的输出shape为（3,2,3,4）。

## 函数原型<a name="zh-cn_topic_0000001265403626_section129451113125413"></a>

```
BROADCAST_INFER(in1_name, in2_name, out_name)
```

该函数会自动调用如下函数：

```
graphStatus BroadCastInfer(const function<vector<int64_t>()> &get_in1_shape,
                           const function<vector<int64_t>()> &get_in2_shape,
                           const function<void(const std::vector<int64_t> &y_shape)> &set_out_shape);
```

## 参数说明<a name="zh-cn_topic_0000001265403626_section1843519321017"></a>

<a name="zh-cn_topic_0000001265403626_table18350576"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265403626_row29692563"><th class="cellrowborder" valign="top" width="27.6%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265403626_p56287378"><a name="zh-cn_topic_0000001265403626_p56287378"></a><a name="zh-cn_topic_0000001265403626_p56287378"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="26.51%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265403626_p62983758"><a name="zh-cn_topic_0000001265403626_p62983758"></a><a name="zh-cn_topic_0000001265403626_p62983758"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="45.89%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265403626_p47160602"><a name="zh-cn_topic_0000001265403626_p47160602"></a><a name="zh-cn_topic_0000001265403626_p47160602"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265403626_row61912443"><td class="cellrowborder" valign="top" width="27.6%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265403626_p1827685185110"><a name="zh-cn_topic_0000001265403626_p1827685185110"></a><a name="zh-cn_topic_0000001265403626_p1827685185110"></a>in1_name</p>
</td>
<td class="cellrowborder" valign="top" width="26.51%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265403626_p13947197203216"><a name="zh-cn_topic_0000001265403626_p13947197203216"></a><a name="zh-cn_topic_0000001265403626_p13947197203216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="45.89%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265403626_p1476526254"><a name="zh-cn_topic_0000001265403626_p1476526254"></a><a name="zh-cn_topic_0000001265403626_p1476526254"></a>算子第一个输入。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265403626_row168464453311"><td class="cellrowborder" valign="top" width="27.6%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265403626_p147558420513"><a name="zh-cn_topic_0000001265403626_p147558420513"></a><a name="zh-cn_topic_0000001265403626_p147558420513"></a>in2_name</p>
</td>
<td class="cellrowborder" valign="top" width="26.51%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265403626_p209461672321"><a name="zh-cn_topic_0000001265403626_p209461672321"></a><a name="zh-cn_topic_0000001265403626_p209461672321"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="45.89%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265403626_p47631661757"><a name="zh-cn_topic_0000001265403626_p47631661757"></a><a name="zh-cn_topic_0000001265403626_p47631661757"></a>算子第二个输入。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265403626_row162281448103111"><td class="cellrowborder" valign="top" width="27.6%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265403626_p1750184856"><a name="zh-cn_topic_0000001265403626_p1750184856"></a><a name="zh-cn_topic_0000001265403626_p1750184856"></a>out_name</p>
</td>
<td class="cellrowborder" valign="top" width="26.51%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265403626_p1712372113211"><a name="zh-cn_topic_0000001265403626_p1712372113211"></a><a name="zh-cn_topic_0000001265403626_p1712372113211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="45.89%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265403626_p3750116055"><a name="zh-cn_topic_0000001265403626_p3750116055"></a><a name="zh-cn_topic_0000001265403626_p3750116055"></a>算子输出。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001265403626_section8491941407"></a>

执行成功或失败。

## 约束说明<a name="zh-cn_topic_0000001265403626_section65498832"></a>

无

## 调用示例<a name="zh-cn_topic_0000001265403626_section78494267595"></a>

```
IMPLEMT_INFERFUNC(RightShift, RightShiftInfer) {
  DataType type = op.GetInputDesc("x").GetDataType();
  SET_OUTPUT_TYPE(op, "z", type);
  return BROADCAST_INFER("x", "y", "z")(op);
}
```

