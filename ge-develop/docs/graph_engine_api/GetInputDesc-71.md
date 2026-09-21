# GetInputDesc<a name="ZH-CN_TOPIC_0000002531200367"></a>

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

## 功能说明<a name="section21477321"></a>

根据算子Input名称或Input索引获取算子Input的TensorDesc。

## 函数原型<a name="section360416384553"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
TensorDesc GetInputDesc(const std::string &name) const
TensorDesc GetInputDescByName(const char_t *name) const
TensorDesc GetInputDesc(uint32_t index) const
```

## 参数说明<a name="section59078165"></a>

<a name="table13077833"></a>
<table><thead align="left"><tr id="row40625737"><th class="cellrowborder" valign="top" width="19.88%" id="mcps1.1.4.1.1"><p id="p2350413"><a name="p2350413"></a><a name="p2350413"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.310000000000002%" id="mcps1.1.4.1.2"><p id="p56165728"><a name="p56165728"></a><a name="p56165728"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="55.81%" id="mcps1.1.4.1.3"><p id="p8575450"><a name="p8575450"></a><a name="p8575450"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row23522831"><td class="cellrowborder" valign="top" width="19.88%" headers="mcps1.1.4.1.1 "><p id="p26301173"><a name="p26301173"></a><a name="p26301173"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="24.310000000000002%" headers="mcps1.1.4.1.2 "><p id="p50020275"><a name="p50020275"></a><a name="p50020275"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="55.81%" headers="mcps1.1.4.1.3 "><p id="p20685786"><a name="p20685786"></a><a name="p20685786"></a>算子Input名称。</p>
<p id="p172064541382"><a name="p172064541382"></a><a name="p172064541382"></a>当无此算子Input名称时，则返回TensorDesc默认构造的对象，其中，主要设置DataType为DT_FLOAT（表示float类型），Format为FORMAT_NCHW（表示NCHW）。</p>
</td>
</tr>
<tr id="row724162013312"><td class="cellrowborder" valign="top" width="19.88%" headers="mcps1.1.4.1.1 "><p id="p1667014535217"><a name="p1667014535217"></a><a name="p1667014535217"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="24.310000000000002%" headers="mcps1.1.4.1.2 "><p id="p26706459529"><a name="p26706459529"></a><a name="p26706459529"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="55.81%" headers="mcps1.1.4.1.3 "><p id="p19670545135211"><a name="p19670545135211"></a><a name="p19670545135211"></a>算子Input索引。</p>
<p id="p167144514522"><a name="p167144514522"></a><a name="p167144514522"></a>当无此算子Input索引时，则返回TensorDesc默认构造的对象，其中，主要设置DataType为DT_FLOAT（表示float类型），Format为FORMAT_NCHW（表示NCHW）。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section61941440"></a>

算子Input的TensorDesc。

## 异常处理<a name="section20602051"></a>

无。

## 约束说明<a name="section51200735"></a>

无。

