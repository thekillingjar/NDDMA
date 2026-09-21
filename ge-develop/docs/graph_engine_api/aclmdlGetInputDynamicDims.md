# aclmdlGetInputDynamicDims<a name="ZH-CN_TOPIC_0000001312641485"></a>

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

## 功能说明<a name="section187304428437"></a>

根据模型描述信息获取模型的输入所支持的动态维度信息。

## 函数原型<a name="section18474205015436"></a>

```
aclError aclmdlGetInputDynamicDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims, size_t gearCount)
```

## 参数说明<a name="section135141717811"></a>

<a name="table1831221142"></a>
<table><thead align="left"><tr id="row3332181419"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p11372191420"><a name="p11372191420"></a><a name="p11372191420"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p13320291417"><a name="p13320291417"></a><a name="p13320291417"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p1940213149"><a name="p1940213149"></a><a name="p1940213149"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row134122191416"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p17403133616514"><a name="p17403133616514"></a><a name="p17403133616514"></a>modelDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1940216361556"><a name="p1940216361556"></a><a name="p1940216361556"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4414215143"><a name="p4414215143"></a><a name="p4414215143"></a>aclmdlDesc类型的指针。</p>
<p id="p35754134454"><a name="p35754134454"></a><a name="p35754134454"></a>需提前调用<a href="aclmdlCreateDesc.md">aclmdlCreateDesc</a>接口创建aclmdlDesc类型的数据。</p>
</td>
</tr>
<tr id="row15525619111910"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p5395036150"><a name="p5395036150"></a><a name="p5395036150"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1539414365517"><a name="p1539414365517"></a><a name="p1539414365517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4273172144416"><a name="p4273172144416"></a><a name="p4273172144416"></a>预留参数，当前未使用，固定设置为-1。</p>
</td>
</tr>
<tr id="row1831212284156"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1962817168218"><a name="p1962817168218"></a><a name="p1962817168218"></a>dims</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p662813166215"><a name="p662813166215"></a><a name="p662813166215"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1162871672120"><a name="p1162871672120"></a><a name="p1162871672120"></a>输入的动态维度信息的指针。</p>
<p id="p424301211459"><a name="p424301211459"></a><a name="p424301211459"></a>dims参数是一个数组，数组中的每个元素指向<a href="aclmdlIODims.md">aclmdlIODims</a>结构，<a href="aclmdlIODims.md">aclmdlIODims</a>结构体中的dims参数也是一个数组，该数组中的每个元素对应每一档中的具体值。</p>
<p id="p4398153710290"><a name="p4398153710290"></a><a name="p4398153710290"></a><strong id="b22814612453"><a name="b22814612453"></a><a name="b22814612453"></a>例如</strong>：</p>
<pre class="screen" id="screen13501001853"><a name="screen13501001853"></a><a name="screen13501001853"></a>aclmdlIODims dims[gearCount];
aclmdlGetInputDynamicDims(model.modelDesc, -1, dims, gearCount);</pre>
</td>
</tr>
<tr id="row218712903311"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1939114361556"><a name="p1939114361556"></a><a name="p1939114361556"></a>gearCount</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p0389123614514"><a name="p0389123614514"></a><a name="p0389123614514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p838703610518"><a name="p838703610518"></a><a name="p838703610518"></a>模型支持的动态维度档位数，需要先通过<a href="aclmdlGetInputDynamicGearCount.md">aclmdlGetInputDynamicGearCount</a>接口获取。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section184151120582"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section6359181812173"></a>

只有在模型构建时设置了动态维度的分档信息后，才可以调用该接口获取动态维度信息。

例如，模型有三个输入，分别为data\(1, 1, 40, -1\)，label\(1, -1\)，mask\(-1, -1\) ， 其中-1表示动态可变。在模型转换时，dynamic\_dims参数的配置示例为：--dynamic\_dims="20,20,1,1;40,40,2,2;80,60,4,4"，则通过本接口获取的动态维度信息为（**[aclmdlIODims](aclmdlIODims.md)**结构体内的name暂不使用）：

-   第0档：
    -   **[aclmdlIODims](aclmdlIODims.md)**结构体内dimCount：8，表示所有输入tensor的维度数量之和
    -   **[aclmdlIODims](aclmdlIODims.md)**结构体内的dims：“1,1,40,20,1,20,1,1”，表示data\(1,1,40,20\)+label\(1,20\)+mask\(1,1\)

-   第1档：
    -   **[aclmdlIODims](aclmdlIODims.md)**结构体内dimCount：8，表示所有输入tensor的维度数量之和
    -   **[aclmdlIODims](aclmdlIODims.md)**结构体内的dims：“1,1,40,40,1,40,2,2”，表示data\(1,1,40,40\)+label\(1,40\)+mask\(2,2\)

-   第2档：
    -   **[aclmdlIODims](aclmdlIODims.md)**结构体内dimCount：8，表示所有输入tensor的维度数量之和
    -   **[aclmdlIODims](aclmdlIODims.md)**结构体内的dims：“1,1,40,80,1,60,4,4”，表示data\(1,1,40,80\)+label\(1,60\)+mask\(4,4\)

