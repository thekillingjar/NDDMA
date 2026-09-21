# GetAllOpName<a name="ZH-CN_TOPIC_0000002516516731"></a>

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

获取Graph中已注册的所有缓存算子的名称列表。

## 函数原型<a name="section1416444114149"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus GetAllOpName(std::vector<std::string> &op_name) const
graphStatus GetAllOpName(std::vector<AscendString> &names) const
```

## 参数说明<a name="section17227446111414"></a>

<a name="table8490113414425"></a>
<table><thead align="left"><tr id="row16490113416427"><th class="cellrowborder" valign="top" width="16.54%" id="mcps1.1.4.1.1"><p id="p3490193424220"><a name="p3490193424220"></a><a name="p3490193424220"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="18.490000000000002%" id="mcps1.1.4.1.2"><p id="p184901234174212"><a name="p184901234174212"></a><a name="p184901234174212"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="64.97%" id="mcps1.1.4.1.3"><p id="p1049013464213"><a name="p1049013464213"></a><a name="p1049013464213"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row249093444210"><td class="cellrowborder" valign="top" width="16.54%" headers="mcps1.1.4.1.1 "><p id="p049073474216"><a name="p049073474216"></a><a name="p049073474216"></a>op_name</p>
</td>
<td class="cellrowborder" valign="top" width="18.490000000000002%" headers="mcps1.1.4.1.2 "><p id="p8490834114211"><a name="p8490834114211"></a><a name="p8490834114211"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="64.97%" headers="mcps1.1.4.1.3 "><p id="p1490103444211"><a name="p1490103444211"></a><a name="p1490103444211"></a>返回Graph中的所有算子的名称。</p>
</td>
</tr>
<tr id="row447310312595"><td class="cellrowborder" valign="top" width="16.54%" headers="mcps1.1.4.1.1 "><p id="p2297153513596"><a name="p2297153513596"></a><a name="p2297153513596"></a>names</p>
</td>
<td class="cellrowborder" valign="top" width="18.490000000000002%" headers="mcps1.1.4.1.2 "><p id="p1629715355594"><a name="p1629715355594"></a><a name="p1629715355594"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="64.97%" headers="mcps1.1.4.1.3 "><p id="p1329715358597"><a name="p1329715358597"></a><a name="p1329715358597"></a>返回Graph中的所有算子的名称。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section92417497143"></a>

<a name="table79991920134013"></a>
<table><thead align="left"><tr id="row69999207406"><th class="cellrowborder" valign="top" width="16.66%" id="mcps1.1.4.1.1"><p id="p59991020114011"><a name="p59991020114011"></a><a name="p59991020114011"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.24%" id="mcps1.1.4.1.2"><p id="p8999720114017"><a name="p8999720114017"></a><a name="p8999720114017"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="64.1%" id="mcps1.1.4.1.3"><p id="p20999220184014"><a name="p20999220184014"></a><a name="p20999220184014"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row299982015406"><td class="cellrowborder" valign="top" width="16.66%" headers="mcps1.1.4.1.1 "><p id="p15999142019406"><a name="p15999142019406"></a><a name="p15999142019406"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.1.4.1.2 "><p id="p17999142004019"><a name="p17999142004019"></a><a name="p17999142004019"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="64.1%" headers="mcps1.1.4.1.3 "><p id="p4999162064013"><a name="p4999162064013"></a><a name="p4999162064013"></a>SUCCESS：操作成功</p>
<p id="p399919209407"><a name="p399919209407"></a><a name="p399919209407"></a>FAILED：操作失败</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1625682141119"></a>

此接口为非必需接口，与[AddOp](AddOp.md)搭配使用。

