# BuildGraph<a name="ZH-CN_TOPIC_0000001312485885"></a>

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

## 功能说明<a name="section44282627"></a>

同步编译指定ID对应的Graph图，生成可用于执行的模型。

## 函数原型<a name="section1831611148519"></a>

```
Status BuildGraph(uint32_t graph_id, const std::vector<InputTensorInfo> &inputs)
Status BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="14.030000000000001%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.59%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.38%" id="mcps1.1.4.1.3"><p id="p16088345"><a name="p16088345"></a><a name="p16088345"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="14.030000000000001%" headers="mcps1.1.4.1.1 "><p id="p6011995"><a name="p6011995"></a><a name="p6011995"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="11.59%" headers="mcps1.1.4.1.2 "><p id="p17209562"><a name="p17209562"></a><a name="p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.38%" headers="mcps1.1.4.1.3 "><p id="p270712592519"><a name="p270712592519"></a><a name="p270712592519"></a>子图对应的ID。</p>
</td>
</tr>
<tr id="row0115745088"><td class="cellrowborder" valign="top" width="14.030000000000001%" headers="mcps1.1.4.1.1 "><p id="p153312242411"><a name="p153312242411"></a><a name="p153312242411"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="11.59%" headers="mcps1.1.4.1.2 "><p id="p12294033221"><a name="p12294033221"></a><a name="p12294033221"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.38%" headers="mcps1.1.4.1.3 "><p id="p38625773518"><a name="p38625773518"></a><a name="p38625773518"></a>当前子图对应的输入数据。</p>
<p id="p253362219247"><a name="p253362219247"></a><a name="p253362219247"></a>InputTensorInfo结构体定义请参见<a href="InputTensorInfo.md">InputTensorInfo</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="13.819999999999999%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.389999999999999%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="73.79%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="13.819999999999999%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.389999999999999%" headers="mcps1.1.4.1.2 "><p id="p1896616536411"><a name="p1896616536411"></a><a name="p1896616536411"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="73.79%" headers="mcps1.1.4.1.3 "><p id="p141243178210"><a name="p141243178210"></a><a name="p141243178210"></a>SUCCESS：编译子图成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：编译子图失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

