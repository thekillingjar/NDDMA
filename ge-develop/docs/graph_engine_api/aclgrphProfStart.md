# aclgrphProfStart<a name="ZH-CN_TOPIC_0000001312645897"></a>

## 产品支持情况<a name="section1993214622115"></a>

<a name="zh-cn_topic_0000001265245842_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265245842_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001265245842_p1883113061818"><a name="zh-cn_topic_0000001265245842_p1883113061818"></a><a name="zh-cn_topic_0000001265245842_p1883113061818"></a><span id="zh-cn_topic_0000001265245842_ph20833205312295"><a name="zh-cn_topic_0000001265245842_ph20833205312295"></a><a name="zh-cn_topic_0000001265245842_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001265245842_p783113012187"><a name="zh-cn_topic_0000001265245842_p783113012187"></a><a name="zh-cn_topic_0000001265245842_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265245842_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265245842_p48327011813"><a name="zh-cn_topic_0000001265245842_p48327011813"></a><a name="zh-cn_topic_0000001265245842_p48327011813"></a><span id="zh-cn_topic_0000001265245842_ph583230201815"><a name="zh-cn_topic_0000001265245842_ph583230201815"></a><a name="zh-cn_topic_0000001265245842_ph583230201815"></a><term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265245842_p108715341013"><a name="zh-cn_topic_0000001265245842_p108715341013"></a><a name="zh-cn_topic_0000001265245842_p108715341013"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265245842_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265245842_p14832120181815"><a name="zh-cn_topic_0000001265245842_p14832120181815"></a><a name="zh-cn_topic_0000001265245842_p14832120181815"></a><span id="zh-cn_topic_0000001265245842_ph1483216010188"><a name="zh-cn_topic_0000001265245842_ph1483216010188"></a><a name="zh-cn_topic_0000001265245842_ph1483216010188"></a><term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265245842_p19948143911820"><a name="zh-cn_topic_0000001265245842_p19948143911820"></a><a name="zh-cn_topic_0000001265245842_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <ge/ge\_prof.h\>
-   库文件：libmsprofiler.so

## 功能说明<a name="section54293220"></a>

下发Profiling请求，使能对应数据的采集。

## 函数原型<a name="section18876936"></a>

```
Status aclgrphProfStart(aclgrphProfConfig *profiler_config)
```

## 参数说明<a name="section35674700"></a>

<a name="table65660028"></a>
<table><thead align="left"><tr id="row36303720"><th class="cellrowborder" valign="top" width="16.96%" id="mcps1.1.4.1.1"><p id="p54920213"><a name="p54920213"></a><a name="p54920213"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.19%" id="mcps1.1.4.1.2"><p id="p19352287"><a name="p19352287"></a><a name="p19352287"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.85%" id="mcps1.1.4.1.3"><p id="p24031405"><a name="p24031405"></a><a name="p24031405"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row386792"><td class="cellrowborder" valign="top" width="16.96%" headers="mcps1.1.4.1.1 "><p id="p31330167"><a name="p31330167"></a><a name="p31330167"></a>profiler_config</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.1.4.1.2 "><p id="p54715587"><a name="p54715587"></a><a name="p54715587"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.4.1.3 "><p id="p2777523"><a name="p2777523"></a><a name="p2777523"></a>Profiling配置信息结构指针。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section52636849"></a>

<a name="table11548462"></a>
<table><thead align="left"><tr id="row19429767"><th class="cellrowborder" valign="top" width="16.99%" id="mcps1.1.4.1.1"><p id="p30307258"><a name="p30307258"></a><a name="p30307258"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.76%" id="mcps1.1.4.1.2"><p id="p38968871"><a name="p38968871"></a><a name="p38968871"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="69.25%" id="mcps1.1.4.1.3"><p id="p2361972"><a name="p2361972"></a><a name="p2361972"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row57102025"><td class="cellrowborder" valign="top" width="16.99%" headers="mcps1.1.4.1.1 "><p id="p61861327"><a name="p61861327"></a><a name="p61861327"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.76%" headers="mcps1.1.4.1.2 "><p id="p44711597"><a name="p44711597"></a><a name="p44711597"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="69.25%" headers="mcps1.1.4.1.3 "><p id="p64869644"><a name="p64869644"></a><a name="p64869644"></a>SUCCESS：成功</p>
<p id="p46955892"><a name="p46955892"></a><a name="p46955892"></a>其他：失败</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section3969601"></a>

-   该接口在[RunGraph](RunGraph.md)之前调用，若在模型执行过程中调用，Profiling采集到的数据为调用aclgrphProfStart接口之后的数据，可能导致数据不完整。
-   该接口和aclgrphProfStop配对使用，先调用aclgrphProfStart接口再调用aclgrphProfStop接口。
-   aclgrphProfInit \> aclgrphProfStart \> aclgrphProfStop \> aclgrphProfFinalize为一条完整的接口调用流程，如果用户想要单进程内切换模型或图，做多轮执行的时候，也需要按照上述完整的流程执行，不支持某个接口的无序调用或并排多次调用。

