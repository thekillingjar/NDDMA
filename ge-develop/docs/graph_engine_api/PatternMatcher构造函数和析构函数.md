# PatternMatcher构造函数和析构函数<a name="ZH-CN_TOPIC_0000002516516765"></a>

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

## 头文件/库文件<a name="section1416611281157"></a>

-   头文件：\#include <ge/fusion/pattern\_matcher.h\>
-   库文件：libge\_compiler.so

## 功能说明<a name="section15686020"></a>

通过Pattern和Target Graph构造一个PatternMatcher。

## 函数原型<a name="section1610715016501"></a>

```
explicit PatternMatcher(std::unique_ptr<Pattern> pattern, const GraphPtr &target_graph)
~PatternMatcher()
```

## 参数说明<a name="section6956456"></a>

<a name="table33761356"></a>
<table><thead align="left"><tr id="row27598891"><th class="cellrowborder" valign="top" width="13.389999999999999%" id="mcps1.1.4.1.1"><p id="p20917673"><a name="p20917673"></a><a name="p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.01%" id="mcps1.1.4.1.2"><p id="p16609919"><a name="p16609919"></a><a name="p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="76.6%" id="mcps1.1.4.1.3"><p id="p59995477"><a name="p59995477"></a><a name="p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row27795493"><td class="cellrowborder" valign="top" width="13.389999999999999%" headers="mcps1.1.4.1.1 "><p id="p443611111822"><a name="p443611111822"></a><a name="p443611111822"></a>pattern</p>
</td>
<td class="cellrowborder" valign="top" width="10.01%" headers="mcps1.1.4.1.2 "><p id="p343614116220"><a name="p343614116220"></a><a name="p343614116220"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.6%" headers="mcps1.1.4.1.3 "><p id="p13435141115210"><a name="p13435141115210"></a><a name="p13435141115210"></a>Pattern定义。构造完PatternMatcher后其生命周期由Matcher管理。</p>
</td>
</tr>
<tr id="row136355457517"><td class="cellrowborder" valign="top" width="13.389999999999999%" headers="mcps1.1.4.1.1 "><p id="p663564510518"><a name="p663564510518"></a><a name="p663564510518"></a>target_graph</p>
</td>
<td class="cellrowborder" valign="top" width="10.01%" headers="mcps1.1.4.1.2 "><p id="p116356452515"><a name="p116356452515"></a><a name="p116356452515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.6%" headers="mcps1.1.4.1.3 "><p id="p166353451514"><a name="p166353451514"></a><a name="p166353451514"></a>待匹配的Graph。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section62608109"></a>

无

## 约束说明<a name="section38092103"></a>

无

