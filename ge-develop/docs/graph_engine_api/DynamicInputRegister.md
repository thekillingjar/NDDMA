# DynamicInputRegister<a name="ZH-CN_TOPIC_0000002499360462"></a>

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

算子动态输入注册。

## 函数原型<a name="section15990165911124"></a>

```
void DynamicInputRegister(const char_t *name, const uint32_t num, bool is_push_back = true)
void DynamicInputRegister(const char_t *name, const uint32_t num, const char_t *datatype_symbol, bool is_push_back = true)
```

## 参数说明<a name="section62506663"></a>

<a name="table34416674"></a>
<table><thead align="left"><tr id="row4508766"><th class="cellrowborder" valign="top" width="33.46%" id="mcps1.1.4.1.1"><p id="p29665754"><a name="p29665754"></a><a name="p29665754"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.37%" id="mcps1.1.4.1.2"><p id="p54115834"><a name="p54115834"></a><a name="p54115834"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="55.169999999999995%" id="mcps1.1.4.1.3"><p id="p48097351"><a name="p48097351"></a><a name="p48097351"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row3571397"><td class="cellrowborder" valign="top" width="33.46%" headers="mcps1.1.4.1.1 "><p id="p20847751"><a name="p20847751"></a><a name="p20847751"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="11.37%" headers="mcps1.1.4.1.2 "><p id="p10946294"><a name="p10946294"></a><a name="p10946294"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="55.169999999999995%" headers="mcps1.1.4.1.3 "><p id="p147851711143915"><a name="p147851711143915"></a><a name="p147851711143915"></a>算子的动态输入名。</p>
</td>
</tr>
<tr id="row37451910229"><td class="cellrowborder" valign="top" width="33.46%" headers="mcps1.1.4.1.1 "><p id="p20751519192215"><a name="p20751519192215"></a><a name="p20751519192215"></a>num</p>
</td>
<td class="cellrowborder" valign="top" width="11.37%" headers="mcps1.1.4.1.2 "><p id="p3832155615225"><a name="p3832155615225"></a><a name="p3832155615225"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="55.169999999999995%" headers="mcps1.1.4.1.3 "><p id="p47591910226"><a name="p47591910226"></a><a name="p47591910226"></a>添加的动态输入个数。</p>
</td>
</tr>
<tr id="row12602162114222"><td class="cellrowborder" valign="top" width="33.46%" headers="mcps1.1.4.1.1 "><p id="p1060292152210"><a name="p1060292152210"></a><a name="p1060292152210"></a>datatype_symbol</p>
</td>
<td class="cellrowborder" valign="top" width="11.37%" headers="mcps1.1.4.1.2 "><p id="p1283811564222"><a name="p1283811564222"></a><a name="p1283811564222"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="55.169999999999995%" headers="mcps1.1.4.1.3 "><p id="p1960292192211"><a name="p1960292192211"></a><a name="p1960292192211"></a>动态输入的数据类型。</p>
</td>
</tr>
<tr id="row1276118309229"><td class="cellrowborder" valign="top" width="33.46%" headers="mcps1.1.4.1.1 "><p id="p5761630192211"><a name="p5761630192211"></a><a name="p5761630192211"></a>is_push_back</p>
</td>
<td class="cellrowborder" valign="top" width="11.37%" headers="mcps1.1.4.1.2 "><p id="p11841125622217"><a name="p11841125622217"></a><a name="p11841125622217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="55.169999999999995%" headers="mcps1.1.4.1.3 "><a name="ul15298143814233"></a><a name="ul15298143814233"></a><ul id="ul15298143814233"><li>true表示在尾部追加动态输入。</li><li>false表示在头部追加动态输入。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25689059"></a>

无。

## 异常处理<a name="section29874939"></a>

无。

## 约束说明<a name="section439002"></a>

无。

