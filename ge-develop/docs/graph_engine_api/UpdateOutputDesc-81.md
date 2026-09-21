# UpdateOutputDesc<a name="ZH-CN_TOPIC_0000002531200377"></a>

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

## 功能说明<a name="section31291646"></a>

根据算子Output名称更新Output的TensorDesc。

## 函数原型<a name="section14460321195712"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus UpdateOutputDesc(const std::string &name, const TensorDesc &tensor_desc)
graphStatus UpdateOutputDesc(const char_t *name, const TensorDesc &tensor_desc)
graphStatus UpdateOutputDesc(const uint32_t index, const TensorDesc &tensor_desc)
```

## 参数说明<a name="section13189358"></a>

<a name="table32576140"></a>
<table><thead align="left"><tr id="row60665573"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="p14964341"><a name="p14964341"></a><a name="p14964341"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.83%" id="mcps1.1.4.1.2"><p id="p4152081"><a name="p4152081"></a><a name="p4152081"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="47.54%" id="mcps1.1.4.1.3"><p id="p62718864"><a name="p62718864"></a><a name="p62718864"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row47063234"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p54025633"><a name="p54025633"></a><a name="p54025633"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="24.83%" headers="mcps1.1.4.1.2 "><p id="p14000148"><a name="p14000148"></a><a name="p14000148"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.54%" headers="mcps1.1.4.1.3 "><p id="p50048984"><a name="p50048984"></a><a name="p50048984"></a>算子Output名称。</p>
</td>
</tr>
<tr id="row47787675"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p45596481"><a name="p45596481"></a><a name="p45596481"></a>tensor_desc</p>
</td>
<td class="cellrowborder" valign="top" width="24.83%" headers="mcps1.1.4.1.2 "><p id="p2327487"><a name="p2327487"></a><a name="p2327487"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.54%" headers="mcps1.1.4.1.3 "><p id="p36936550"><a name="p36936550"></a><a name="p36936550"></a>TensorDesc对象。</p>
</td>
</tr>
<tr id="row132335217548"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p144838555417"><a name="p144838555417"></a><a name="p144838555417"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="24.83%" headers="mcps1.1.4.1.2 "><p id="p1548319516545"><a name="p1548319516545"></a><a name="p1548319516545"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="47.54%" headers="mcps1.1.4.1.3 "><p id="p12483359549"><a name="p12483359549"></a><a name="p12483359549"></a>算子Output的序号。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section51595365"></a>

graphStatus类型：更新TensorDesc成功，返回GRAPH\_SUCCESS，否则，返回GRAPH\_FAILED。

## 异常处理<a name="section61705107"></a>

无。

## 约束说明<a name="section18475057"></a>

无。

