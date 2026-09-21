# TryGetInputDesc<a name="ZH-CN_TOPIC_0000002499360480"></a>

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

## 功能说明<a name="section23101557"></a>

根据算子Input名称获取算子Input的TensorDesc。

## 函数原型<a name="section45525325619"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus TryGetInputDesc(const std::string &name, TensorDesc &tensor_desc) const
graphStatus TryGetInputDesc(const char_t *name, TensorDesc &tensor_desc) const
```

## 参数说明<a name="section6587426"></a>

<a name="table27978103"></a>
<table><thead align="left"><tr id="row36709949"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="p20715931"><a name="p20715931"></a><a name="p20715931"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="23.75%" id="mcps1.1.4.1.2"><p id="p268861"><a name="p268861"></a><a name="p268861"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="48.620000000000005%" id="mcps1.1.4.1.3"><p id="p19171695"><a name="p19171695"></a><a name="p19171695"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9403471"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p23483667"><a name="p23483667"></a><a name="p23483667"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="23.75%" headers="mcps1.1.4.1.2 "><p id="p23128868"><a name="p23128868"></a><a name="p23128868"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="48.620000000000005%" headers="mcps1.1.4.1.3 "><p id="p15361866"><a name="p15361866"></a><a name="p15361866"></a>算子的Input名。</p>
</td>
</tr>
<tr id="row4039073"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p58729529"><a name="p58729529"></a><a name="p58729529"></a>tensor_desc</p>
</td>
<td class="cellrowborder" valign="top" width="23.75%" headers="mcps1.1.4.1.2 "><p id="p59471393"><a name="p59471393"></a><a name="p59471393"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="48.620000000000005%" headers="mcps1.1.4.1.3 "><p id="p20875522"><a name="p20875522"></a><a name="p20875522"></a>返回算子端口的当前设置格式，为TensorDesc对象。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section59286834"></a>

graphStatus类型：True，有此端口，获取TensorDesc成功；False，无此端口，出参为空，获取TensorDesc失败。

## 异常处理<a name="section63819464"></a>

<a name="table21044820"></a>
<table><thead align="left"><tr id="row20764673"><th class="cellrowborder" valign="top" width="27.200000000000003%" id="mcps1.1.3.1.1"><p id="p4216941"><a name="p4216941"></a><a name="p4216941"></a>异常场景</p>
</th>
<th class="cellrowborder" valign="top" width="72.8%" id="mcps1.1.3.1.2"><p id="p6027901"><a name="p6027901"></a><a name="p6027901"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row18497945"><td class="cellrowborder" valign="top" width="27.200000000000003%" headers="mcps1.1.3.1.1 "><p id="p21938602"><a name="p21938602"></a><a name="p21938602"></a>无对应name输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.8%" headers="mcps1.1.3.1.2 "><p id="p32196376"><a name="p32196376"></a><a name="p32196376"></a>返回False。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section37504268"></a>

无。

