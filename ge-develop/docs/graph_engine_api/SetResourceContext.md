# SetResourceContext<a name="ZH-CN_TOPIC_0000002531280397"></a>

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

## 功能说明<a name="zh-cn_topic_0000001265084814_section5719164819393"></a>

为标识为key的资源，设置资源上下文对象，并交由资源上下文管理器管理。

此接口一般由写类型的资源类算子调用，如stack push等。

## 函数原型<a name="zh-cn_topic_0000001265084814_section129319456392"></a>

```
graphStatus SetResourceContext(const ge::AscendString &key, ResourceContext *resource_context)
```

## 参数说明<a name="zh-cn_topic_0000001265084814_section1618815223918"></a>

<a name="zh-cn_topic_0000001265084814_table1692051423212"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265084814_row292051453218"><th class="cellrowborder" valign="top" width="28.07%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265084814_p18920141417322"><a name="zh-cn_topic_0000001265084814_p18920141417322"></a><a name="zh-cn_topic_0000001265084814_p18920141417322"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="16.46%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265084814_p392051443210"><a name="zh-cn_topic_0000001265084814_p392051443210"></a><a name="zh-cn_topic_0000001265084814_p392051443210"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="55.47%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265084814_p1992091443212"><a name="zh-cn_topic_0000001265084814_p1992091443212"></a><a name="zh-cn_topic_0000001265084814_p1992091443212"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265084814_row13920171493213"><td class="cellrowborder" valign="top" width="28.07%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265084814_p193241318133212"><a name="zh-cn_topic_0000001265084814_p193241318133212"></a><a name="zh-cn_topic_0000001265084814_p193241318133212"></a>key</p>
</td>
<td class="cellrowborder" valign="top" width="16.46%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265084814_p17323111833214"><a name="zh-cn_topic_0000001265084814_p17323111833214"></a><a name="zh-cn_topic_0000001265084814_p17323111833214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="55.47%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265084814_p153211018183220"><a name="zh-cn_topic_0000001265084814_p153211018183220"></a><a name="zh-cn_topic_0000001265084814_p153211018183220"></a>资源唯一标识。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265084814_row3366533154912"><td class="cellrowborder" valign="top" width="28.07%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265084814_p143671433144917"><a name="zh-cn_topic_0000001265084814_p143671433144917"></a><a name="zh-cn_topic_0000001265084814_p143671433144917"></a>resource_context</p>
</td>
<td class="cellrowborder" valign="top" width="16.46%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265084814_p173671338494"><a name="zh-cn_topic_0000001265084814_p173671338494"></a><a name="zh-cn_topic_0000001265084814_p173671338494"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="55.47%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265084814_p1367133317491"><a name="zh-cn_topic_0000001265084814_p1367133317491"></a><a name="zh-cn_topic_0000001265084814_p1367133317491"></a>资源上下文对象指针，可参见<a href="GetResourceContext.md">GetResourceContext</a>接口的返回值说明。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001265084814_section12573195503920"></a>

graphStatus类型：GRAPH\_SUCCESS，代表成功；GRAPH\_FAILED，代表失败。

## 约束说明<a name="zh-cn_topic_0000001265084814_section1286618413401"></a>

若使用[Create](Create.md)接口创建InferenceContext时未传入resource context管理器指针，则该接口返回失败。

