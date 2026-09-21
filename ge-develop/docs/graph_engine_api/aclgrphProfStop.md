# aclgrphProfStop<a name="ZH-CN_TOPIC_0000001265404738"></a>

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

## 功能说明<a name="section13136003"></a>

停止Profiling数据采集。

## 函数原型<a name="section51115164"></a>

```
Status aclgrphProfStop(aclgrphProfConfig *profiler_config)
```

## 参数说明<a name="section57383300"></a>

<a name="table18021669"></a>
<table><thead align="left"><tr id="row12436382"><th class="cellrowborder" valign="top" width="16.2%" id="mcps1.1.4.1.1"><p id="p714004"><a name="p714004"></a><a name="p714004"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.44%" id="mcps1.1.4.1.2"><p id="p57834400"><a name="p57834400"></a><a name="p57834400"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.36%" id="mcps1.1.4.1.3"><p id="p54074791"><a name="p54074791"></a><a name="p54074791"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17981971"><td class="cellrowborder" valign="top" width="16.2%" headers="mcps1.1.4.1.1 "><p id="p47253580"><a name="p47253580"></a><a name="p47253580"></a>profiler_config</p>
</td>
<td class="cellrowborder" valign="top" width="12.44%" headers="mcps1.1.4.1.2 "><p id="p2334777"><a name="p2334777"></a><a name="p2334777"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.1.4.1.3 "><p id="p54899267"><a name="p54899267"></a><a name="p54899267"></a>Profiling配置信息结构指针。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section46687659"></a>

<a name="table24682973"></a>
<table><thead align="left"><tr id="row43067999"><th class="cellrowborder" valign="top" width="16.41%" id="mcps1.1.4.1.1"><p id="p65955856"><a name="p65955856"></a><a name="p65955856"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.01%" id="mcps1.1.4.1.2"><p id="p40824114"><a name="p40824114"></a><a name="p40824114"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="70.58%" id="mcps1.1.4.1.3"><p id="p18418924"><a name="p18418924"></a><a name="p18418924"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row15537895"><td class="cellrowborder" valign="top" width="16.41%" headers="mcps1.1.4.1.1 "><p id="p50609995"><a name="p50609995"></a><a name="p50609995"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.01%" headers="mcps1.1.4.1.2 "><p id="p5768894"><a name="p5768894"></a><a name="p5768894"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="70.58%" headers="mcps1.1.4.1.3 "><p id="p64627307"><a name="p64627307"></a><a name="p64627307"></a>SUCCESS：成功</p>
<p id="p44774854"><a name="p44774854"></a><a name="p44774854"></a>其他：失败</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section17535755"></a>

该接口和[aclgrphProfStart](aclgrphProfStart.md)配对使用，先调用aclgrphProfStart接口再调用aclgrphProfStop接口。

