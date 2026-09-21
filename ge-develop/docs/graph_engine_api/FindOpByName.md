# FindOpByName<a name="ZH-CN_TOPIC_0000002516516729"></a>

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

## 功能说明<a name="section1182557201116"></a>

基于算子名称，获取缓存在Graph中的op对象。

## 函数原型<a name="section1963913110115"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus FindOpByName(const std::string &name, ge::Operator &op) const
graphStatus FindOpByName(const char_t *name, ge::Operator &op) const
```

## 参数说明<a name="section7128141161112"></a>

<a name="table33761356"></a>
<table><thead align="left"><tr id="row27598891"><th class="cellrowborder" valign="top" width="15.690000000000001%" id="mcps1.1.4.1.1"><p id="p20917673"><a name="p20917673"></a><a name="p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.42%" id="mcps1.1.4.1.2"><p id="p16609919"><a name="p16609919"></a><a name="p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.89%" id="mcps1.1.4.1.3"><p id="p59995477"><a name="p59995477"></a><a name="p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row27795493"><td class="cellrowborder" valign="top" width="15.690000000000001%" headers="mcps1.1.4.1.1 "><p id="p36842478"><a name="p36842478"></a><a name="p36842478"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="14.42%" headers="mcps1.1.4.1.2 "><p id="p31450711"><a name="p31450711"></a><a name="p31450711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.89%" headers="mcps1.1.4.1.3 "><p id="p55467050"><a name="p55467050"></a><a name="p55467050"></a>需要获取的算子名称。</p>
</td>
</tr>
<tr id="row1878062193714"><td class="cellrowborder" valign="top" width="15.690000000000001%" headers="mcps1.1.4.1.1 "><p id="p12781152114376"><a name="p12781152114376"></a><a name="p12781152114376"></a>op</p>
</td>
<td class="cellrowborder" valign="top" width="14.42%" headers="mcps1.1.4.1.2 "><p id="p5781182113717"><a name="p5781182113717"></a><a name="p5781182113717"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.89%" headers="mcps1.1.4.1.3 "><p id="p17781192114379"><a name="p17781192114379"></a><a name="p17781192114379"></a>返回用户所需要的op对象。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2207121510113"></a>

<a name="table1051417571311"></a>
<table><thead align="left"><tr id="row3515195711311"><th class="cellrowborder" valign="top" width="16.16%" id="mcps1.1.4.1.1"><p id="p155156577319"><a name="p155156577319"></a><a name="p155156577319"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.580000000000002%" id="mcps1.1.4.1.2"><p id="p35151578314"><a name="p35151578314"></a><a name="p35151578314"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="69.26%" id="mcps1.1.4.1.3"><p id="p19515657103111"><a name="p19515657103111"></a><a name="p19515657103111"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row351555763118"><td class="cellrowborder" valign="top" width="16.16%" headers="mcps1.1.4.1.1 "><p id="p3515175763117"><a name="p3515175763117"></a><a name="p3515175763117"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.580000000000002%" headers="mcps1.1.4.1.2 "><p id="p145863073910"><a name="p145863073910"></a><a name="p145863073910"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="69.26%" headers="mcps1.1.4.1.3 "><p id="p1049243914395"><a name="p1049243914395"></a><a name="p1049243914395"></a>SUCCESS：成功获取算子实例。</p>
<p id="p5921154116382"><a name="p5921154116382"></a><a name="p5921154116382"></a>FAILED：此名字没有在Graph中注册op对象。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1625682141119"></a>

此接口为非必须接口，与[AddOp](AddOp.md)搭配使用。

