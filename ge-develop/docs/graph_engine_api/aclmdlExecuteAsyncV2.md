# aclmdlExecuteAsyncV2<a name="ZH-CN_TOPIC_0000001569833200"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p13650147112412"><a name="p13650147112412"></a><a name="p13650147112412"></a>☓</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p265210476246"><a name="p265210476246"></a><a name="p265210476246"></a>☓</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section36583473819"></a>

根据[aclmdlSetExecConfigOpt](aclmdlSetExecConfigOpt.md)所配置的属性，执行模型推理，直到返回推理结果。该接口是在[aclmdlExecuteAsync](aclmdlExecuteAsync.md)接口基础上进行了增强，支持在执行模型推理时控制一些配置参数。异步接口。

本接口需要配合其它接口一起使用，实现模型执行，接口调用顺序如下：

1.  调用[aclmdlCreateExecConfigHandle](aclmdlCreateExecConfigHandle.md)接口创建模型执行的配置对象。
2.  多次调用[aclmdlSetExecConfigOpt](aclmdlSetExecConfigOpt.md)接口设置配置对象中每个属性的值。
3.  调用aclmdlExecuteAsyncV2接口指定模型执行时需要的配置信息，并进行模型执行。
4.  模型执行成功后，调用[aclmdlDestroyExecConfigHandle](aclmdlDestroyExecConfigHandle.md)接口销毁。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclmdlExecuteAsyncV2(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream, const aclmdlExecConfigHandle *handle)
```

## 参数说明<a name="section75395119104"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>modelId</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p16390135183518"><a name="p16390135183518"></a><a name="p16390135183518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4388859358"><a name="p4388859358"></a><a name="p4388859358"></a>指定需要执行推理的模型的ID。</p>
<p id="p57291851112517"><a name="p57291851112517"></a><a name="p57291851112517"></a>调用<a href="aclmdlLoadFromFile.md">aclmdlLoadFromFile</a>接口/<a href="aclmdlLoadFromMem.md">aclmdlLoadFromMem</a>接口/<a href="aclmdlLoadFromFileWithMem.md">aclmdlLoadFromFileWithMem</a>接口/<a href="aclmdlLoadFromMemWithMem.md">aclmdlLoadFromMemWithMem</a>接口加载模型成功后，会返回模型ID。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1291021213420"><a name="p1291021213420"></a><a name="p1291021213420"></a>input</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p7910212173413"><a name="p7910212173413"></a><a name="p7910212173413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p491091220346"><a name="p491091220346"></a><a name="p491091220346"></a>模型推理的输入数据的指针。</p>
</td>
</tr>
<tr id="row137987158341"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p13798191516347"><a name="p13798191516347"></a><a name="p13798191516347"></a>output</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p17798815103410"><a name="p17798815103410"></a><a name="p17798815103410"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p2798415163418"><a name="p2798415163418"></a><a name="p2798415163418"></a>模型推理的输出数据的指针。</p>
</td>
</tr>
<tr id="row229015351354"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1129193583512"><a name="p1129193583512"></a><a name="p1129193583512"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1329163512353"><a name="p1329163512353"></a><a name="p1329163512353"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p329163563513"><a name="p329163563513"></a><a name="p329163563513"></a>指定Stream。</p>
</td>
</tr>
<tr id="row5515544193619"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p151554453617"><a name="p151554453617"></a><a name="p151554453617"></a>handle</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p251517440364"><a name="p251517440364"></a><a name="p251517440364"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p481255414379"><a name="p481255414379"></a><a name="p481255414379"></a>模型执行的配置对象的指针。与<a href="aclmdlSetExecConfigOpt.md">aclmdlSetExecConfigOpt</a>中的handle保持一致。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section96219408320"></a>

约束与[aclmdlExecuteAsync](aclmdlExecuteAsync.md)一致。

