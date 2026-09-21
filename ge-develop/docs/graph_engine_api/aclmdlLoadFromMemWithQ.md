# aclmdlLoadFromMemWithQ<a name="ZH-CN_TOPIC_0000001265400678"></a>

## 产品支持情况<a name="section8178181118225"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section1041322555815"></a>

从内存加载离线模型数据（适配昇腾AI处理器的离线模型），模型的输入、输出数据都存放在队列中。本接口只支持加载固定Shape输入的模型。

## 函数原型<a name="section4937192885810"></a>

```
aclError aclmdlLoadFromMemWithQ(const void *model, size_t modelSize, uint32_t *modelId, const uint32_t *inputQ, size_t inputQNum, const uint32_t *outputQ, size_t outputQNum)
```

## 参数说明<a name="section19228533155815"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p16390135183518"><a name="p16390135183518"></a><a name="p16390135183518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4388859358"><a name="p4388859358"></a><a name="p4388859358"></a>存放模型数据的内存地址指针。</p>
</td>
</tr>
<tr id="row193678191464"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1291021213420"><a name="p1291021213420"></a><a name="p1291021213420"></a>modelSize</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p7910212173413"><a name="p7910212173413"></a><a name="p7910212173413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p64351912165911"><a name="p64351912165911"></a><a name="p64351912165911"></a>内存中的模型数据长度，单位Byte。</p>
</td>
</tr>
<tr id="row1263820333519"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p13798191516347"><a name="p13798191516347"></a><a name="p13798191516347"></a>modelId</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p17798815103410"><a name="p17798815103410"></a><a name="p17798815103410"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1037012182337"><a name="p1037012182337"></a><a name="p1037012182337"></a>模型ID的指针。</p>
<p id="p8851114185610"><a name="p8851114185610"></a><a name="p8851114185610"></a>系统成功加载模型后，返回模型ID作为后续操作时识别模型的标志。</p>
</td>
</tr>
<tr id="row133583695114"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p10364368517"><a name="p10364368517"></a><a name="p10364368517"></a>inputQ</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p63616362518"><a name="p63616362518"></a><a name="p63616362518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p73603615519"><a name="p73603615519"></a><a name="p73603615519"></a>队列ID的指针，一个模型的输入对应一个队列ID。</p>
</td>
</tr>
<tr id="row1990725185218"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1990225135220"><a name="p1990225135220"></a><a name="p1990225135220"></a>inputQNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p2990202517524"><a name="p2990202517524"></a><a name="p2990202517524"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p199111259528"><a name="p199111259528"></a><a name="p199111259528"></a>输入队列大小。</p>
</td>
</tr>
<tr id="row11236162817526"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p13236122820524"><a name="p13236122820524"></a><a name="p13236122820524"></a>outputQ</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1323672845218"><a name="p1323672845218"></a><a name="p1323672845218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p89737563563"><a name="p89737563563"></a><a name="p89737563563"></a>队列ID的指针，一个模型的输出对应一个队列ID。</p>
</td>
</tr>
<tr id="row1235853215215"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p0358193245211"><a name="p0358193245211"></a><a name="p0358193245211"></a>outputQNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p235803285210"><a name="p235803285210"></a><a name="p235803285210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p14973656125616"><a name="p14973656125616"></a><a name="p14973656125616"></a>输出队列大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section941963755812"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section2676619153310"></a>

模型加载、模型执行、模型卸载的操作必须在同一个Context下（关于Context的创建请参见aclrtSetDevice或aclrtCreateContext）。在加载前，请先根据模型文件的大小评估内存空间是否足够，内存空间不足，会导致应用程序异常。

## 参考资源<a name="section1490215432573"></a>

当前还提供了[aclmdlSetConfigOpt](aclmdlSetConfigOpt.md)接口、[aclmdlLoadWithConfig](aclmdlLoadWithConfig.md)接口来实现模型加载，通过配置对象中的属性来区分，在加载模型时是从文件加载，还是从内存加载，以及内存是由系统内部管理，还是由用户管理。

