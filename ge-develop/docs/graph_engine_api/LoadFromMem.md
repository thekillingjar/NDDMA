# LoadFromMem<a name="ZH-CN_TOPIC_0000002516356747"></a>

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

-   头文件：\#include <graph/graph.h\>
-   库文件：libgraph.so

## 功能说明<a name="section145571943181415"></a>

从内存中读取序列化后Graph。

## 函数原型<a name="section1416444114149"></a>

```
graphStatus LoadFromMem(const GraphBuffer &graph_buffer)
graphStatus LoadFromMem(const uint8_t *data, const size_t len)
```

## 参数说明<a name="section17227446111414"></a>

<a name="table8490113414425"></a>
<table><thead align="left"><tr id="row16490113416427"><th class="cellrowborder" valign="top" width="18.34%" id="mcps1.1.4.1.1"><p id="p3490193424220"><a name="p3490193424220"></a><a name="p3490193424220"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.78%" id="mcps1.1.4.1.2"><p id="p184901234174212"><a name="p184901234174212"></a><a name="p184901234174212"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="66.88%" id="mcps1.1.4.1.3"><p id="p1049013464213"><a name="p1049013464213"></a><a name="p1049013464213"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row249093444210"><td class="cellrowborder" valign="top" width="18.34%" headers="mcps1.1.4.1.1 "><p id="p83014916422"><a name="p83014916422"></a><a name="p83014916422"></a>graph_buffer</p>
</td>
<td class="cellrowborder" valign="top" width="14.78%" headers="mcps1.1.4.1.2 "><p id="p8490834114211"><a name="p8490834114211"></a><a name="p8490834114211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.88%" headers="mcps1.1.4.1.3 "><p id="p1490103444211"><a name="p1490103444211"></a><a name="p1490103444211"></a>读取的Graph的buffer。</p>
</td>
</tr>
<tr id="row116291024112116"><td class="cellrowborder" valign="top" width="18.34%" headers="mcps1.1.4.1.1 "><p id="p3631724172113"><a name="p3631724172113"></a><a name="p3631724172113"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="14.78%" headers="mcps1.1.4.1.2 "><p id="p499109102319"><a name="p499109102319"></a><a name="p499109102319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.88%" headers="mcps1.1.4.1.3 "><p id="p66314240218"><a name="p66314240218"></a><a name="p66314240218"></a>读取的Graph的内存起始位置。</p>
</td>
</tr>
<tr id="row136331726122119"><td class="cellrowborder" valign="top" width="18.34%" headers="mcps1.1.4.1.1 "><p id="p2063322672112"><a name="p2063322672112"></a><a name="p2063322672112"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="14.78%" headers="mcps1.1.4.1.2 "><p id="p1199989162313"><a name="p1199989162313"></a><a name="p1199989162313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.88%" headers="mcps1.1.4.1.3 "><p id="p863392614211"><a name="p863392614211"></a><a name="p863392614211"></a>读取的Graph的内存长度。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section92417497143"></a>

<a name="table1051417571311"></a>
<table><thead align="left"><tr id="row3515195711311"><th class="cellrowborder" valign="top" width="18.42%" id="mcps1.1.4.1.1"><p id="p155156577319"><a name="p155156577319"></a><a name="p155156577319"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.55%" id="mcps1.1.4.1.2"><p id="p35151578314"><a name="p35151578314"></a><a name="p35151578314"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="66.03%" id="mcps1.1.4.1.3"><p id="p19515657103111"><a name="p19515657103111"></a><a name="p19515657103111"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row351555763118"><td class="cellrowborder" valign="top" width="18.42%" headers="mcps1.1.4.1.1 "><p id="p3515175763117"><a name="p3515175763117"></a><a name="p3515175763117"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.1.4.1.2 "><p id="p145863073910"><a name="p145863073910"></a><a name="p145863073910"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="66.03%" headers="mcps1.1.4.1.3 "><p id="p1049243914395"><a name="p1049243914395"></a><a name="p1049243914395"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="p54441032123711"><a name="p54441032123711"></a><a name="p54441032123711"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section20470355181419"></a>

仅支持读取air格式的文件。

