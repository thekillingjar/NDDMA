# GetVariables<a name="ZH-CN_TOPIC_0000001312725933"></a>

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

-   头文件：\#include <ge/ge\_api.h\>
-   库文件：libge\_runner.so

## 功能说明<a name="section61382201"></a>

变量查询接口，获取Session内所有variable算子或指定variable算子的Tensor内容。

## 函数原型<a name="section15568903"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
Status GetVariables(const std::vector<std::string> &var_names, std::vector<Tensor> &var_values)
Status GetVariables(const std::vector<AscendString> &var_names, std::vector<Tensor> &var_values)
```

## 参数说明<a name="section5902405"></a>

<a name="table19987085"></a>
<table><thead align="left"><tr id="row32738149"><th class="cellrowborder" valign="top" width="14.39%" id="mcps1.1.4.1.1"><p id="p34544438"><a name="p34544438"></a><a name="p34544438"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.35%" id="mcps1.1.4.1.2"><p id="p46636132"><a name="p46636132"></a><a name="p46636132"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.26%" id="mcps1.1.4.1.3"><p id="p19430358"><a name="p19430358"></a><a name="p19430358"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row30355189"><td class="cellrowborder" valign="top" width="14.39%" headers="mcps1.1.4.1.1 "><p id="p3967718182718"><a name="p3967718182718"></a><a name="p3967718182718"></a>var_names</p>
</td>
<td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.2 "><p id="p159661618132713"><a name="p159661618132713"></a><a name="p159661618132713"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.26%" headers="mcps1.1.4.1.3 "><p id="p1396614186271"><a name="p1396614186271"></a><a name="p1396614186271"></a>variable算子的名字，为空时返回所有variable类型算子的值。</p>
</td>
</tr>
<tr id="row50297679"><td class="cellrowborder" valign="top" width="14.39%" headers="mcps1.1.4.1.1 "><p id="p1596681892712"><a name="p1596681892712"></a><a name="p1596681892712"></a>var_values</p>
</td>
<td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.2 "><p id="p1965101812279"><a name="p1965101812279"></a><a name="p1965101812279"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.26%" headers="mcps1.1.4.1.3 "><p id="p2965818112713"><a name="p2965818112713"></a><a name="p2965818112713"></a>variable算子的值，按照tensor格式返回；与var_names保持一致序列。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section53121653"></a>

<a name="table60309786"></a>
<table><thead align="left"><tr id="row33911763"><th class="cellrowborder" valign="top" width="14.42%" id="mcps1.1.4.1.1"><p id="p62498284"><a name="p62498284"></a><a name="p62498284"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.509999999999998%" id="mcps1.1.4.1.2"><p id="p29196278"><a name="p29196278"></a><a name="p29196278"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="73.07000000000001%" id="mcps1.1.4.1.3"><p id="p16088345"><a name="p16088345"></a><a name="p16088345"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row28087548"><td class="cellrowborder" valign="top" width="14.42%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.509999999999998%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="73.07000000000001%" headers="mcps1.1.4.1.3 "><p id="p27314278552"><a name="p27314278552"></a><a name="p27314278552"></a>SUCCESS：成功</p>
<p id="p6731127135519"><a name="p6731127135519"></a><a name="p6731127135519"></a>其他：不存在该name的variable算子；variable未执行初始化等场景。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section8332830"></a>

无

