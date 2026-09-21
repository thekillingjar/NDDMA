# GetDynamicOutputDesc<a name="ZH-CN_TOPIC_0000002531280411"></a>

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

## 功能说明<a name="section63390070"></a>

根据name和index的组合获取算子动态Output的TensorDesc。

## 函数原型<a name="section72247955"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
TensorDesc GetDynamicOutputDesc(const std::string &name, uint32_t index) const
TensorDesc GetDynamicOutputDesc(const char_t *name, uint32_t index) const
```

## 参数说明<a name="section33639718"></a>

<a name="table44853923"></a>
<table><thead align="left"><tr id="row32789387"><th class="cellrowborder" valign="top" width="22.6%" id="mcps1.1.4.1.1"><p id="p38694667"><a name="p38694667"></a><a name="p38694667"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="21.310000000000002%" id="mcps1.1.4.1.2"><p id="p47260332"><a name="p47260332"></a><a name="p47260332"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="56.089999999999996%" id="mcps1.1.4.1.3"><p id="p32089967"><a name="p32089967"></a><a name="p32089967"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row49150562"><td class="cellrowborder" valign="top" width="22.6%" headers="mcps1.1.4.1.1 "><p id="p21772565"><a name="p21772565"></a><a name="p21772565"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="21.310000000000002%" headers="mcps1.1.4.1.2 "><p id="p18747312"><a name="p18747312"></a><a name="p18747312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="56.089999999999996%" headers="mcps1.1.4.1.3 "><p id="p13471132114116"><a name="p13471132114116"></a><a name="p13471132114116"></a>算子动态Output的名称。</p>
</td>
</tr>
<tr id="row49347015"><td class="cellrowborder" valign="top" width="22.6%" headers="mcps1.1.4.1.1 "><p id="p37685299"><a name="p37685299"></a><a name="p37685299"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="21.310000000000002%" headers="mcps1.1.4.1.2 "><p id="p32610412"><a name="p32610412"></a><a name="p32610412"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="56.089999999999996%" headers="mcps1.1.4.1.3 "><p id="p2567181019414"><a name="p2567181019414"></a><a name="p2567181019414"></a>算子动态Output编号，编号从1开始。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section34322009"></a>

获取TensorDesc成功，则返回算子动态Output的TensorDesc；获取失败，则返回TensorDesc默认构造的对象，其中，主要设置DataType为DT\_FLOAT（表示float类型），Format为FORMAT\_NCHW（表示NCHW）。

## 异常处理<a name="section40462633"></a>

无。

## 约束说明<a name="section28619382"></a>

无。

