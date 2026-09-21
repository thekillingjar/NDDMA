# GetOutDataNodesAndPortIndexs<a name="ZH-CN_TOPIC_0000002484276828"></a>

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

获取算子指定输出端口对应的对端算子以及对端算子的输入端口号。

## 函数原型<a name="zh-cn_topic_0235751031_section1610715016501"></a>

```
std::vector<std::pair<GNodePtr, int32_t>> GetOutDataNodesAndPortIndexs(const int32_t index) const
```

## 参数说明<a name="zh-cn_topic_0235751031_section6956456"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.82%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.38%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0235751031_row27795493"><td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.1 "><p id="p11969557325"><a name="p11969557325"></a><a name="p11969557325"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="15.82%" headers="mcps1.1.4.1.2 "><p id="p11908236142716"><a name="p11908236142716"></a><a name="p11908236142716"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.38%" headers="mcps1.1.4.1.3 "><p id="p186056225818"><a name="p186056225818"></a><a name="p186056225818"></a>算子的输出端口号。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0235751031_section62608109"></a>

<a name="zh-cn_topic_0235751031_table2601186"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row1832323"><th class="cellrowborder" valign="top" width="7.430000000000001%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p14200498"><a name="zh-cn_topic_0235751031_p14200498"></a><a name="zh-cn_topic_0235751031_p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="23.549999999999997%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p9389685"><a name="zh-cn_topic_0235751031_p9389685"></a><a name="zh-cn_topic_0235751031_p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="69.02000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p22367029"><a name="zh-cn_topic_0235751031_p22367029"></a><a name="zh-cn_topic_0235751031_p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0235751031_row66898905"><td class="cellrowborder" valign="top" width="7.430000000000001%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0235751031_p50102218"><a name="zh-cn_topic_0235751031_p50102218"></a><a name="zh-cn_topic_0235751031_p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="23.549999999999997%" headers="mcps1.1.4.1.2 "><p id="p10349134617483"><a name="p10349134617483"></a><a name="p10349134617483"></a>vector&lt;std::pair&lt;GNodePtr, int32_t &gt;&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="69.02000000000001%" headers="mcps1.1.4.1.3 "><p id="p14175322125818"><a name="p14175322125818"></a><a name="p14175322125818"></a>获取算子指定输出端口对应的对端算子以及对端算子的输入端口号</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section15328459144813"></a>

无

