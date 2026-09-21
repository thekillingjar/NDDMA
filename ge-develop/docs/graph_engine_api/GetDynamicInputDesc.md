# GetDynamicInputDesc<a name="ZH-CN_TOPIC_0000002499520440"></a>

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

## 功能说明<a name="section17792110"></a>

根据name和index的组合获取算子动态Input的TensorDesc。

## 函数原型<a name="section1143634485719"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
TensorDesc GetDynamicInputDesc(const std::string &name, uint32_t index) const
TensorDesc GetDynamicInputDesc(const char_t *name, uint32_t index) const
```

## 参数说明<a name="section25911262"></a>

<a name="table44962972"></a>
<table><thead align="left"><tr id="row39638585"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="p56608836"><a name="p56608836"></a><a name="p56608836"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.22%" id="mcps1.1.4.1.2"><p id="p21913016"><a name="p21913016"></a><a name="p21913016"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="48.15%" id="mcps1.1.4.1.3"><p id="p24111608"><a name="p24111608"></a><a name="p24111608"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row6883221"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p20670010"><a name="p20670010"></a><a name="p20670010"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="24.22%" headers="mcps1.1.4.1.2 "><p id="p63658128"><a name="p63658128"></a><a name="p63658128"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.15%" headers="mcps1.1.4.1.3 "><p id="p13471132114116"><a name="p13471132114116"></a><a name="p13471132114116"></a>算子动态Input的名称。</p>
</td>
</tr>
<tr id="row47144157"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p60580335"><a name="p60580335"></a><a name="p60580335"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="24.22%" headers="mcps1.1.4.1.2 "><p id="p8060118"><a name="p8060118"></a><a name="p8060118"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.15%" headers="mcps1.1.4.1.3 "><p id="p2567181019414"><a name="p2567181019414"></a><a name="p2567181019414"></a>算子动态Input编号，编号从0开始。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section31874772"></a>

获取TensorDesc成功，则返回算子动态Input的TensorDesc；获取失败，则返回TensorDesc默认构造的对象，其中，主要设置DataType为DT\_FLOAT（表示float类型），Format为FORMAT\_NCHW（表示NCHW）。

## 异常处理<a name="section18437496"></a>

无。

## 约束说明<a name="section31719739"></a>

无。

