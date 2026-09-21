# AddControlInput<a name="ZH-CN_TOPIC_0000002499520434"></a>

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

## 功能说明<a name="section93821533102312"></a>

添加算子的控制边，控制边目前只是控制算子的执行顺序。

## 函数原型<a name="section1323919702319"></a>

```
Operator &AddControlInput(const Operator &src_oprt)
```

## 参数说明<a name="section6622193032717"></a>

<a name="table91371237195617"></a>
<table><thead align="left"><tr id="row1913743775619"><th class="cellrowborder" valign="top" width="21.88%" id="mcps1.1.4.1.1"><p id="p51371737105614"><a name="p51371737105614"></a><a name="p51371737105614"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.049999999999999%" id="mcps1.1.4.1.2"><p id="p15137237175617"><a name="p15137237175617"></a><a name="p15137237175617"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="63.07000000000001%" id="mcps1.1.4.1.3"><p id="p613803712564"><a name="p613803712564"></a><a name="p613803712564"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row8138183712564"><td class="cellrowborder" valign="top" width="21.88%" headers="mcps1.1.4.1.1 "><p id="p8138037115611"><a name="p8138037115611"></a><a name="p8138037115611"></a>src_oprt</p>
</td>
<td class="cellrowborder" valign="top" width="15.049999999999999%" headers="mcps1.1.4.1.2 "><p id="p191385370562"><a name="p191385370562"></a><a name="p191385370562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="63.07000000000001%" headers="mcps1.1.4.1.3 "><p id="p1013818371566"><a name="p1013818371566"></a><a name="p1013818371566"></a>控制边对应的源算子。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section947464642719"></a>

算子对象本身。

## 异常处理<a name="section11969151184218"></a>

无

## 约束说明<a name="section29637165284"></a>

无

