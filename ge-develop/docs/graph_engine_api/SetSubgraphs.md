# SetSubgraphs<a name="ZH-CN_TOPIC_0000002520381075"></a>

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

-   头文件：\#include <graph/gnode.h\>
-   库文件：libgraph.so

## 功能说明<a name="zh-cn_topic_0235751031_section15686020"></a>

更新指定输出端口的tensor格式。

## 函数原型<a name="zh-cn_topic_0235751031_section1610715016501"></a>

```
graphStatus SetSubgraphs(const AscendString &subgraph_ir_name, const std::vector<Graph> &subgraphs)
```

## 参数说明<a name="zh-cn_topic_0235751031_section6956456"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.89%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="66.57%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0235751031_row27795493"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p281116107213"><a name="p281116107213"></a><a name="p281116107213"></a>subgraph_ir_name</p>
</td>
<td class="cellrowborder" valign="top" width="14.89%" headers="mcps1.1.4.1.2 "><p id="p1581312105219"><a name="p1581312105219"></a><a name="p1581312105219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.57%" headers="mcps1.1.4.1.3 "><p id="p186056225818"><a name="p186056225818"></a><a name="p186056225818"></a>算子IR注册时的动态个数的子图名称，以.DYNAMIC_GRAPH("Dx")的注册方式举例，subgraph_ir_name对应"Dx"</p>
</td>
</tr>
<tr id="row16125121361"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.1.4.1.1 "><p id="p772720154214"><a name="p772720154214"></a><a name="p772720154214"></a>subgraphs</p>
</td>
<td class="cellrowborder" valign="top" width="14.89%" headers="mcps1.1.4.1.2 "><p id="p2612151212364"><a name="p2612151212364"></a><a name="p2612151212364"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.57%" headers="mcps1.1.4.1.3 "><p id="p3612212173619"><a name="p3612212173619"></a><a name="p3612212173619"></a>要绑定节点上的多个子图实例</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0235751031_section62608109"></a>

<a name="zh-cn_topic_0235751031_table2601186"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row1832323"><th class="cellrowborder" valign="top" width="19.24%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p14200498"><a name="zh-cn_topic_0235751031_p14200498"></a><a name="zh-cn_topic_0235751031_p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.75%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p9389685"><a name="zh-cn_topic_0235751031_p9389685"></a><a name="zh-cn_topic_0235751031_p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="66.01%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p22367029"><a name="zh-cn_topic_0235751031_p22367029"></a><a name="zh-cn_topic_0235751031_p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0235751031_row66898905"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0235751031_p50102218"><a name="zh-cn_topic_0235751031_p50102218"></a><a name="zh-cn_topic_0235751031_p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="14.75%" headers="mcps1.1.4.1.2 "><p id="p73026092813"><a name="p73026092813"></a><a name="p73026092813"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="66.01%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0235751031_p918814449444"><a name="zh-cn_topic_0235751031_p918814449444"></a><a name="zh-cn_topic_0235751031_p918814449444"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="zh-cn_topic_0235751031_p91883446443"><a name="zh-cn_topic_0235751031_p91883446443"></a><a name="zh-cn_topic_0235751031_p91883446443"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0235751031_section38092103"></a>

无

