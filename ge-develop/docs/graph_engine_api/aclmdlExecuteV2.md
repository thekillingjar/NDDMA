# aclmdlExecuteV2<a name="ZH-CN_TOPIC_0000001597420421"></a>

## 产品支持情况<a name="section16107182283615"></a>

<a name="zh-cn_topic_0000002219420921_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002219420921_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002219420921_p1883113061818"><a name="zh-cn_topic_0000002219420921_p1883113061818"></a><a name="zh-cn_topic_0000002219420921_p1883113061818"></a><span id="zh-cn_topic_0000002219420921_ph20833205312295"><a name="zh-cn_topic_0000002219420921_ph20833205312295"></a><a name="zh-cn_topic_0000002219420921_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002219420921_p783113012187"><a name="zh-cn_topic_0000002219420921_p783113012187"></a><a name="zh-cn_topic_0000002219420921_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002219420921_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p48327011813"><a name="zh-cn_topic_0000002219420921_p48327011813"></a><a name="zh-cn_topic_0000002219420921_p48327011813"></a><span id="zh-cn_topic_0000002219420921_ph583230201815"><a name="zh-cn_topic_0000002219420921_ph583230201815"></a><a name="zh-cn_topic_0000002219420921_ph583230201815"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p7948163910184"><a name="zh-cn_topic_0000002219420921_p7948163910184"></a><a name="zh-cn_topic_0000002219420921_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002219420921_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p14832120181815"><a name="zh-cn_topic_0000002219420921_p14832120181815"></a><a name="zh-cn_topic_0000002219420921_p14832120181815"></a><span id="zh-cn_topic_0000002219420921_ph1483216010188"><a name="zh-cn_topic_0000002219420921_ph1483216010188"></a><a name="zh-cn_topic_0000002219420921_ph1483216010188"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p19948143911820"><a name="zh-cn_topic_0000002219420921_p19948143911820"></a><a name="zh-cn_topic_0000002219420921_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section36583473819"></a>

根据[aclmdlSetExecConfigOpt](aclmdlSetExecConfigOpt.md)接口所配置的属性，执行模型推理，直到返回推理结果。该接口是在[aclmdlExecute](aclmdlExecute.md)接口基础上进行了增强，支持在执行模型推理时控制一些配置参数。

本接口需要配合其它接口一起使用，实现模型执行，接口调用顺序如下：

1.  调用[aclmdlCreateExecConfigHandle](aclmdlCreateExecConfigHandle.md)接口创建模型执行的配置对象。
2.  多次调用[aclmdlSetExecConfigOpt](aclmdlSetExecConfigOpt.md)接口设置配置对象中每个属性的值。
3.  调用aclmdlExecuteV2接口指定模型执行时需要的配置信息，并进行模型执行。
4.  模型执行成功后，调用[aclmdlDestroyExecConfigHandle](aclmdlDestroyExecConfigHandle.md)接口销毁。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclmdlExecuteV2(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream, const aclmdlExecConfigHandle *handle)
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
<p id="p14166627124315"><a name="p14166627124315"></a><a name="p14166627124315"></a>调用模型加载接口（例如<a href="aclmdlLoadFromFile.md">aclmdlLoadFromFile</a>接口、<a href="aclmdlLoadFromMem.md">aclmdlLoadFromMem</a>等）成功后，会返回模型ID，该ID作为本接口的输入。</p>
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
<p id="p384615205486"><a name="p384615205486"></a><a name="p384615205486"></a>调用aclCreateDataBuffer接口创建存放对应index<strong id="zh-cn_topic_0000001264921926_b5695182122512"><a name="zh-cn_topic_0000001264921926_b5695182122512"></a><a name="zh-cn_topic_0000001264921926_b5695182122512"></a>输出数据</strong>的aclDataBuffer类型时，<strong id="zh-cn_topic_0000001264921926_b17204124472514"><a name="zh-cn_topic_0000001264921926_b17204124472514"></a><a name="zh-cn_topic_0000001264921926_b17204124472514"></a>支持在data参数处传入nullptr，同时size需设置为0</strong>，表示创建一个空的aclDataBuffer类型，然后在模型执行过程中，系统<strong id="zh-cn_topic_0000001264921926_b890112719283"><a name="zh-cn_topic_0000001264921926_b890112719283"></a><a name="zh-cn_topic_0000001264921926_b890112719283"></a>内部自行计算并申请</strong>该index输出的内存。使用该方式可节省内存，但内存数据使用结束后，需由用户释放内存并重置aclDataBuffer，同时，系统内部申请内存时涉及内存拷贝，可能涉及性能损耗。</p>
<div class="p" id="p10395124017015"><a name="p10395124017015"></a><a name="p10395124017015"></a>释放内存并重置aclDataBuffer的示例代码如下：<a name="zh-cn_topic_0000001264921926_screen29751447185416"></a><a name="zh-cn_topic_0000001264921926_screen29751447185416"></a><pre class="screen" codetype="Cpp" id="zh-cn_topic_0000001264921926_screen29751447185416">aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output, 0); // 根据index获取对应的dataBuffer        
void *data = aclGetDataBufferAddr(dataBuffer);  // 获取data的Device指针
aclrtFree(data ); // 释放Device内存
aclUpdateDataBuffer(dataBuffer, nullptr, 0); // 重置dataBuffer里面内容，以便下次推理</pre>
</div>
</td>
</tr>
<tr id="row229015351354"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1129193583512"><a name="p1129193583512"></a><a name="p1129193583512"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1329163512353"><a name="p1329163512353"></a><a name="p1329163512353"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p329163563513"><a name="p329163563513"></a><a name="p329163563513"></a>指定Stream。</p>
<p id="p13619192010348"><a name="p13619192010348"></a><a name="p13619192010348"></a>若此处传NULL，则通过<a href="aclmdlSetExecConfigOpt.md">aclmdlSetExecConfigOpt</a>接口配置的属性值不生效。</p>
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

约束与[aclmdlExecute](aclmdlExecute.md)一致。

