# SetStreamId<a name="ZH-CN_TOPIC_0000002516356803"></a>

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

## 功能说明<a name="section865990310"></a>

设置某节点的Stream ID，拥有相同Stream ID的节点将会在同一条流上依次执行。

## 函数原型<a name="section36595013115"></a>

```
graphStatus SetStreamId(const GNode &node, int64_t stream_id)
```

## 参数说明<a name="section66599010110"></a>

<a name="table146591702113"></a>
<table><thead align="left"><tr id="row9660703111"><th class="cellrowborder" valign="top" width="12.94%" id="mcps1.1.4.1.1"><p id="p1566050316"><a name="p1566050316"></a><a name="p1566050316"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.1.4.1.2"><p id="p11660303111"><a name="p11660303111"></a><a name="p11660303111"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.01%" id="mcps1.1.4.1.3"><p id="p766015011111"><a name="p766015011111"></a><a name="p766015011111"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row566020312"><td class="cellrowborder" valign="top" width="12.94%" headers="mcps1.1.4.1.1 "><p id="p205416461114"><a name="p205416461114"></a><a name="p205416461114"></a>node</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p6660110413"><a name="p6660110413"></a><a name="p6660110413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.01%" headers="mcps1.1.4.1.3 "><p id="p1183133016303"><a name="p1183133016303"></a><a name="p1183133016303"></a>图上节点。</p>
</td>
</tr>
<tr id="row137911391096"><td class="cellrowborder" valign="top" width="12.94%" headers="mcps1.1.4.1.1 "><p id="p2037903917920"><a name="p2037903917920"></a><a name="p2037903917920"></a>stream_id</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p03809397913"><a name="p03809397913"></a><a name="p03809397913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.01%" headers="mcps1.1.4.1.3 "><p id="p23803392912"><a name="p23803392912"></a><a name="p23803392912"></a>待设置的Stream ID：若为已申请的stream id，直接设置即可；若需要新申请Stream ID，请先调用<a href="AllocateNextStreamId.md">AllocateNextStreamId</a>接口申请，否则若Stream ID超出当前图上最大的Stream ID接口将返回失败。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section666040417"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="13.639999999999999%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.67%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.1.4.1.2 "><p id="p1896616536411"><a name="p1896616536411"></a><a name="p1896616536411"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="76.67%" headers="mcps1.1.4.1.3 "><p id="p141243178210"><a name="p141243178210"></a><a name="p141243178210"></a>SUCCESS：设置成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：设置失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section3659801615"></a>

无

