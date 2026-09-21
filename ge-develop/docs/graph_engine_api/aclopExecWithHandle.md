# aclopExecWithHandle<a name="ZH-CN_TOPIC_0000001265400746"></a>

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

## 功能说明<a name="section97607407416"></a>

以Handle方式调用一个算子，不支持动态Shape算子，动态Shape算子请使用[aclopExecuteV2](aclopExecuteV2.md)。异步接口。

## 函数原型<a name="section09265441348"></a>

```
aclError aclopExecWithHandle(aclopHandle *handle,
int numInputs,
const aclDataBuffer *const inputs[],
int numOutputs,
aclDataBuffer *const outputs[],
aclrtStream stream)
```

## 参数说明<a name="section1858610511420"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0122830089_p1088611422254"><a name="zh-cn_topic_0122830089_p1088611422254"></a><a name="zh-cn_topic_0122830089_p1088611422254"></a>handle</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p112297500513"><a name="p112297500513"></a><a name="p112297500513"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1712295213116"><a name="p1712295213116"></a><a name="p1712295213116"></a>算子执行handle的指针。</p>
<p id="p172536289565"><a name="p172536289565"></a><a name="p172536289565"></a>需提前调用<a href="aclopCreateHandle.md">aclopCreateHandle</a>接口创建aclopHandle类型的数据。</p>
</td>
</tr>
<tr id="row186711285514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p16868328115110"><a name="p16868328115110"></a><a name="p16868328115110"></a>numInputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p148684283518"><a name="p148684283518"></a><a name="p148684283518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p212018526117"><a name="p212018526117"></a><a name="p212018526117"></a>算子输入tensor的数量。</p>
</td>
</tr>
<tr id="row85323116518"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p125418315514"><a name="p125418315514"></a><a name="p125418315514"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p354173165116"><a name="p354173165116"></a><a name="p354173165116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p61171652213"><a name="p61171652213"></a><a name="p61171652213"></a>算子输入tensor的指针数组。</p>
<p id="p2175227165719"><a name="p2175227165719"></a><a name="p2175227165719"></a>需提前调用<span id="ph0151456133520"><a name="ph0151456133520"></a><a name="ph0151456133520"></a>aclCreateDataBuffer</span>接口创建aclDataBuffer类型的数据。</p>
<p id="p133679599519"><a name="p133679599519"></a><a name="p133679599519"></a>inputs数组中的元素个数必须与numInputs参数值保持一致。</p>
</td>
</tr>
<tr id="row6739833145111"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1740933185118"><a name="p1740933185118"></a><a name="p1740933185118"></a>numOutputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p19740433115115"><a name="p19740433115115"></a><a name="p19740433115115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p12410154317110"><a name="p12410154317110"></a><a name="p12410154317110"></a>算子输出tensor的数量。</p>
</td>
</tr>
<tr id="row2325183745112"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1325163719511"><a name="p1325163719511"></a><a name="p1325163719511"></a>outputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p15325437165114"><a name="p15325437165114"></a><a name="p15325437165114"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1154712459119"><a name="p1154712459119"></a><a name="p1154712459119"></a>算子输出tensor的指针数组。</p>
<p id="p645630141419"><a name="p645630141419"></a><a name="p645630141419"></a>需提前调用<span id="ph159754380392"><a name="ph159754380392"></a><a name="ph159754380392"></a>aclCreateDataBuffer</span>接口创建aclDataBuffer类型的数据。</p>
<p id="p1263710144529"><a name="p1263710144529"></a><a name="p1263710144529"></a>outputs数组中的元素个数必须与numOutputs参数值保持一致</p>
</td>
</tr>
<tr id="row27321339165112"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p11732193955117"><a name="p11732193955117"></a><a name="p11732193955117"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p473223915512"><a name="p473223915512"></a><a name="p473223915512"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p14900547314"><a name="p14900547314"></a><a name="p14900547314"></a>该算子需要加载的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section13189155513418"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section162871491166"></a>

-   多线程场景下，不支持调用本接口时指定同一个Stream或使用默认Stream，否则可能任务执行异常。
-   执行有可选输入的算子时，如果可选输入不使用，则需按此种方式创建aclDataBuffer类型的数据：aclCreateDataBuffer\(nullptr, 0\)，同时aclDataBuffer中的数据不需要释放，因为是空指针。

