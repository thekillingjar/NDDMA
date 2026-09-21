# Create<a name="ZH-CN_TOPIC_0000002499520426"></a>

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

## 功能说明<a name="zh-cn_topic_0000001312724813_section5719164819393"></a>

在资源类算子推理的上下文中，创建资源算子的上下文对象。

## 函数原型<a name="zh-cn_topic_0000001312724813_section129319456392"></a>

```
static std::unique_ptr<InferenceContext> Create(
void *resource_context_mgr = nullptr
)
```

## 参数说明<a name="zh-cn_topic_0000001312724813_section1618815223918"></a>

<a name="zh-cn_topic_0000001312724813_table15371192192515"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312724813_row103715232512"><th class="cellrowborder" valign="top" width="22.57%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312724813_p1537162162513"><a name="zh-cn_topic_0000001312724813_p1537162162513"></a><a name="zh-cn_topic_0000001312724813_p1537162162513"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.54%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312724813_p8372102182517"><a name="zh-cn_topic_0000001312724813_p8372102182517"></a><a name="zh-cn_topic_0000001312724813_p8372102182517"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="63.89%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312724813_p163721225254"><a name="zh-cn_topic_0000001312724813_p163721225254"></a><a name="zh-cn_topic_0000001312724813_p163721225254"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312724813_row637212192515"><td class="cellrowborder" valign="top" width="22.57%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312724813_p9578617172512"><a name="zh-cn_topic_0000001312724813_p9578617172512"></a><a name="zh-cn_topic_0000001312724813_p9578617172512"></a>resource_context_mgr</p>
</td>
<td class="cellrowborder" valign="top" width="13.54%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312724813_p193727215255"><a name="zh-cn_topic_0000001312724813_p193727215255"></a><a name="zh-cn_topic_0000001312724813_p193727215255"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.89%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312724813_p2072621319256"><a name="zh-cn_topic_0000001312724813_p2072621319256"></a><a name="zh-cn_topic_0000001312724813_p2072621319256"></a>Resource Context管理器指针。</p>
<p id="zh-cn_topic_0000001312724813_p1964710258271"><a name="zh-cn_topic_0000001312724813_p1964710258271"></a><a name="zh-cn_topic_0000001312724813_p1964710258271"></a>Session创建时候会初始化此指针，由InferShape框架自动传入，生命周期同Session。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001312724813_section12573195503920"></a>

资源类算子间传递的上下文对象。

## 异常处理<a name="zh-cn_topic_0000001312724813_section66861205403"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001312724813_section1286618413401"></a>

无。

