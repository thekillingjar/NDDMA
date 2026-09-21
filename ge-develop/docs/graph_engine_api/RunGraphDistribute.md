# RunGraphDistribute<a name="ZH-CN_TOPIC_0000001755703838"></a>

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

对输入refdata节点切分后，同步运行指定ID对应的Graph图，输出运行结果。

与[RunGraph](RunGraph.md)接口的区别是：该接口输入的refdata节点是经过切分后的，输出是每个Device的输出结果。

## 函数原型<a name="section1831611148519"></a>

```
Status RunGraphDistribute(uint32_t graph_id, const std::map<int32_t, std::vector<Tensor>> &device_to_inputs, std::map<int32_t, std::vector<Tensor>> &device_to_outputs)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="11.27%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.16%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.57%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.1.4.1.1 "><p id="p429044010131"><a name="p429044010131"></a><a name="p429044010131"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.2 "><p id="p5700193491914"><a name="p5700193491914"></a><a name="p5700193491914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.57%" headers="mcps1.1.4.1.3 "><p id="p15858163831917"><a name="p15858163831917"></a><a name="p15858163831917"></a>要运行图对应的ID。</p>
</td>
</tr>
<tr id="row0115745088"><td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.1.4.1.1 "><p id="p2286154017136"><a name="p2286154017136"></a><a name="p2286154017136"></a>device_to_inputs</p>
</td>
<td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.2 "><p id="p12700183411913"><a name="p12700183411913"></a><a name="p12700183411913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.57%" headers="mcps1.1.4.1.3 "><p id="p1285853810193"><a name="p1285853810193"></a><a name="p1285853810193"></a>计算图输入Tensor，为Host上分配的内存空间。</p>
<p id="p985873811910"><a name="p985873811910"></a><a name="p985873811910"></a>使用const std::map&lt;int32_t, std::vector&lt;Tensor&gt;&gt;作为输入，为切分后每个Device ID下对应的输入。</p>
</td>
</tr>
<tr id="row7157114920813"><td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.1.4.1.1 "><p id="p93921542111315"><a name="p93921542111315"></a><a name="p93921542111315"></a>device_to_outputs</p>
</td>
<td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.2 "><p id="p177002034181911"><a name="p177002034181911"></a><a name="p177002034181911"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="77.57%" headers="mcps1.1.4.1.3 "><p id="p15859143891918"><a name="p15859143891918"></a><a name="p15859143891918"></a>计算图输出Tensor，用户无需分配内存空间，运行完成后GE会分配内存并赋值。</p>
<p id="p685917386192"><a name="p685917386192"></a><a name="p685917386192"></a>使用std::map&lt;int32_t, std::vector&lt;Tensor&gt;&gt;作为输出结果，记录每个Device ID对应的计算图结果。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.15%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.68%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="77.17%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.15%" headers="mcps1.1.4.1.1 "><p id="p1687718435137"><a name="p1687718435137"></a><a name="p1687718435137"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.68%" headers="mcps1.1.4.1.2 "><p id="p1875343131313"><a name="p1875343131313"></a><a name="p1875343131313"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="77.17%" headers="mcps1.1.4.1.3 "><a name="ul13500593351"></a><a name="ul13500593351"></a><ul id="ul13500593351"><li>SUCCESS：运行子图成功。</li><li>错误码请参考<span id="ph1075716510011"><a name="ph1075716510011"></a><a name="ph1075716510011"></a>《应用开发指南 (C&amp;C++)》</span><span id="ph2355145121019"><a name="ph2355145121019"></a><a name="ph2355145121019"></a></span>。</li></ul>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

-   对于graph\_id中的全量输入，输入顺序约束为：模型data输入 + batch\_index + kv。
-   对于graph\_id中的增量输入，输入顺序约束为：模型data输入 + kv。

