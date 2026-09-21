# AttrRegister<a name="ZH-CN_TOPIC_0000002531200357"></a>

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

## 头文件和库文件<a name="section094612611340"></a>

-   头文件：\#include <graph/operator.h\>
-   库文件：libgraph.so

## 功能说明<a name="section59140967"></a>

泛型属性注册接口。

## 函数原型<a name="section15990165911124"></a>

```
void AttrRegister(const char_t *name, const AttrValue &attr_value)
```

## 参数说明<a name="section62506663"></a>

<a name="table34416674"></a>
<table><thead align="left"><tr id="row4508766"><th class="cellrowborder" valign="top" width="15.120000000000001%" id="mcps1.1.4.1.1"><p id="p29665754"><a name="p29665754"></a><a name="p29665754"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.479999999999999%" id="mcps1.1.4.1.2"><p id="p54115834"><a name="p54115834"></a><a name="p54115834"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.39999999999999%" id="mcps1.1.4.1.3"><p id="p48097351"><a name="p48097351"></a><a name="p48097351"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row3571397"><td class="cellrowborder" valign="top" width="15.120000000000001%" headers="mcps1.1.4.1.1 "><p id="p19437195755910"><a name="p19437195755910"></a><a name="p19437195755910"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="12.479999999999999%" headers="mcps1.1.4.1.2 "><p id="p10946294"><a name="p10946294"></a><a name="p10946294"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.39999999999999%" headers="mcps1.1.4.1.3 "><p id="p20606121120010"><a name="p20606121120010"></a><a name="p20606121120010"></a>属性名。</p>
</td>
</tr>
<tr id="row1738420104"><td class="cellrowborder" valign="top" width="15.120000000000001%" headers="mcps1.1.4.1.1 "><p id="p338514016011"><a name="p338514016011"></a><a name="p338514016011"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="12.479999999999999%" headers="mcps1.1.4.1.2 "><p id="p16385120300"><a name="p16385120300"></a><a name="p16385120300"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.39999999999999%" headers="mcps1.1.4.1.3 "><p id="p133851108010"><a name="p133851108010"></a><a name="p133851108010"></a>属性取值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25689059"></a>

graphStatus类型：成功，返回GRAPH\_SUCCESS，否则，返回GRAPH\_FAILED。

## 异常处理<a name="section29874939"></a>

无

## 约束说明<a name="section439002"></a>

无。

