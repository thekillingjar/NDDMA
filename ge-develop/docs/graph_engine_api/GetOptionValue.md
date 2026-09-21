# GetOptionValue<a name="ZH-CN_TOPIC_0000002530073391"></a>

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

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <register/register\_custom\_pass.h\>
-   库文件：libregister.so

## 功能说明<a name="section18628811541"></a>

通过option的key，从上下文中获取option的值。

## 函数原型<a name="section227616159541"></a>

```
graphStatus GetOptionValue(const AscendString &option_key, AscendString &option_value) const
```

## 参数说明<a name="section19716172412542"></a>

<a name="table13766133358"></a>
<table><thead align="left"><tr id="row7376141310352"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.1.4.1.1"><p id="p113761013123516"><a name="p113761013123516"></a><a name="p113761013123516"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.889999999999999%" id="mcps1.1.4.1.2"><p id="p63761113193514"><a name="p63761113193514"></a><a name="p63761113193514"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.57%" id="mcps1.1.4.1.3"><p id="p183767139354"><a name="p183767139354"></a><a name="p183767139354"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1037691393516"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p12376813113515"><a name="p12376813113515"></a><a name="p12376813113515"></a>option_key</p>
</td>
<td class="cellrowborder" valign="top" width="13.889999999999999%" headers="mcps1.1.4.1.2 "><p id="p193761613103519"><a name="p193761613103519"></a><a name="p193761613103519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.57%" headers="mcps1.1.4.1.3 "><p id="p73664269352"><a name="p73664269352"></a><a name="p73664269352"></a>option的key。</p>
</td>
</tr>
<tr id="row482874316238"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p1682814438234"><a name="p1682814438234"></a><a name="p1682814438234"></a>option_value</p>
</td>
<td class="cellrowborder" valign="top" width="13.889999999999999%" headers="mcps1.1.4.1.2 "><p id="p1682964332317"><a name="p1682964332317"></a><a name="p1682964332317"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.57%" headers="mcps1.1.4.1.3 "><p id="p682912437237"><a name="p682912437237"></a><a name="p682912437237"></a>option的值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section18768183311547"></a>

状态码，若option key不存在，则返回失败。

## 约束说明<a name="section16880122012547"></a>

无

