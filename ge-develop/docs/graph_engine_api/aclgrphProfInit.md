# aclgrphProfInit<a name="ZH-CN_TOPIC_0000001265245858"></a>

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

## 功能说明<a name="section61382201"></a>

初始化Profiling，设置Profiling参数（目前供用户设置保存性能数据文件的路径）。

## 函数原型<a name="section15568903"></a>

```
Status aclgrphProfInit(const char *profiler_path, uint32_t length)
```

## 参数说明<a name="section5902405"></a>

<a name="table19987085"></a>
<table><thead align="left"><tr id="row32738149"><th class="cellrowborder" valign="top" width="13.600000000000001%" id="mcps1.1.4.1.1"><p id="p34544438"><a name="p34544438"></a><a name="p34544438"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.379999999999999%" id="mcps1.1.4.1.2"><p id="p46636132"><a name="p46636132"></a><a name="p46636132"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.02%" id="mcps1.1.4.1.3"><p id="p19430358"><a name="p19430358"></a><a name="p19430358"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row30355189"><td class="cellrowborder" valign="top" width="13.600000000000001%" headers="mcps1.1.4.1.1 "><p id="p42851240"><a name="p42851240"></a><a name="p42851240"></a>profiler_path</p>
</td>
<td class="cellrowborder" valign="top" width="9.379999999999999%" headers="mcps1.1.4.1.2 "><p id="p48398424"><a name="p48398424"></a><a name="p48398424"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.02%" headers="mcps1.1.4.1.3 "><p id="p29997509121"><a name="p29997509121"></a><a name="p29997509121"></a>指定保存性能数据的文件的路径，路径支持绝对路径和相对路径。</p>
</td>
</tr>
<tr id="row50297679"><td class="cellrowborder" valign="top" width="13.600000000000001%" headers="mcps1.1.4.1.1 "><p id="p47580213"><a name="p47580213"></a><a name="p47580213"></a>length</p>
</td>
<td class="cellrowborder" valign="top" width="9.379999999999999%" headers="mcps1.1.4.1.2 "><p id="p28792051"><a name="p28792051"></a><a name="p28792051"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.02%" headers="mcps1.1.4.1.3 "><p id="p50454834"><a name="p50454834"></a><a name="p50454834"></a>profiler_path的长度，单位为字节。最大长度不超过4096字节。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section53121653"></a>

<a name="table60309786"></a>
<table><thead align="left"><tr id="row33911763"><th class="cellrowborder" valign="top" width="13.86%" id="mcps1.1.4.1.1"><p id="p62498284"><a name="p62498284"></a><a name="p62498284"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.15%" id="mcps1.1.4.1.2"><p id="p29196278"><a name="p29196278"></a><a name="p29196278"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.99000000000001%" id="mcps1.1.4.1.3"><p id="p16088345"><a name="p16088345"></a><a name="p16088345"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row28087548"><td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.1.4.1.1 "><p id="p60498887"><a name="p60498887"></a><a name="p60498887"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="9.15%" headers="mcps1.1.4.1.2 "><p id="p1462819"><a name="p1462819"></a><a name="p1462819"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="76.99000000000001%" headers="mcps1.1.4.1.3 "><p id="p51379511"><a name="p51379511"></a><a name="p51379511"></a>SUCCESS：成功。</p>
<p id="p5128167237"><a name="p5128167237"></a><a name="p5128167237"></a>FAILED：失败。</p>
<p id="p158814441538"><a name="p158814441538"></a><a name="p158814441538"></a>ACL_ERROR_FEATURE_UNSUPPORTED：动态Profiling场景下不支持调用aclgrphProfInit接口。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section8332830"></a>

-   不支持多次重复调用aclgrphProfInit，并且该接口需和[aclgrphProfFinalize](aclgrphProfFinalize.md)配对使用，先调用aclgrphProfInit接口再调用aclgrphProfFinalize接口。
-   建议该接口在[GEInitialize](GEInitialize.md)之后，[AddGraph](AddGraph.md)之前被调用，可采集到AddGraph时的Profiling数据。

