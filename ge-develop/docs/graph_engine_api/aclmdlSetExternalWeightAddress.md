# aclmdlSetExternalWeightAddress<a name="ZH-CN_TOPIC_0000002477134930"></a>

## 产品支持情况<a name="section15254644421"></a>

<a name="zh-cn_topic_0000002219420921_table14931115524110"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002219420921_row1993118556414"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002219420921_p29315553419"><a name="zh-cn_topic_0000002219420921_p29315553419"></a><a name="zh-cn_topic_0000002219420921_p29315553419"></a><span id="zh-cn_topic_0000002219420921_ph59311455164119"><a name="zh-cn_topic_0000002219420921_ph59311455164119"></a><a name="zh-cn_topic_0000002219420921_ph59311455164119"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002219420921_p59313557417"><a name="zh-cn_topic_0000002219420921_p59313557417"></a><a name="zh-cn_topic_0000002219420921_p59313557417"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002219420921_row1693117553411"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p1493195513412"><a name="zh-cn_topic_0000002219420921_p1493195513412"></a><a name="zh-cn_topic_0000002219420921_p1493195513412"></a><span id="zh-cn_topic_0000002219420921_ph1093110555418"><a name="zh-cn_topic_0000002219420921_ph1093110555418"></a><a name="zh-cn_topic_0000002219420921_ph1093110555418"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p20931175524111"><a name="zh-cn_topic_0000002219420921_p20931175524111"></a><a name="zh-cn_topic_0000002219420921_p20931175524111"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002219420921_row199312559416"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p0931555144119"><a name="zh-cn_topic_0000002219420921_p0931555144119"></a><a name="zh-cn_topic_0000002219420921_p0931555144119"></a><span id="zh-cn_topic_0000002219420921_ph1693115559411"><a name="zh-cn_topic_0000002219420921_ph1693115559411"></a><a name="zh-cn_topic_0000002219420921_ph1693115559411"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p129321955154117"><a name="zh-cn_topic_0000002219420921_p129321955154117"></a><a name="zh-cn_topic_0000002219420921_p129321955154117"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section36583473819"></a>

配置存放外置权重的Device内存，调用方需在模型加载前将外置权重数据复制到该内存中。若存在多个外置权重文件，则需多次调用此接口。

模型加载过程中，计算图中的每一个FileConstant节点会保存对应外置权重文件的路径（例如/xx/../weight\_hasid1），图引擎（ Graph Engine ，简称GE）根据路径中的文件名（例如weight\_hasid1）检索用户是否配置了外置权重的Device内存。若已配置，则直接使用该Device内存地址；否则，GE将自行申请一块Device内存，并读取外置权重文件，并将外置权重数据拷贝至该Device内存中，模型卸载时会释放该内存。

![](figures/Device-Context-Stream之间的关系.png)

**本接口需要与以下其它接口配合**，实现模型加载功能：

1.  调用[aclmdlCreateConfigHandle](aclmdlCreateConfigHandle.md)接口创建模型加载的配置对象。
2.  （可选）调用[aclmdlSetExternalWeightAddress](aclmdlSetExternalWeightAddress.md)接口配置存放外置权重的Device内存。
3.  多次调用[aclmdlSetConfigOpt](aclmdlSetConfigOpt.md)接口设置配置对象中每个属性的值。
4.  调用[aclmdlLoadWithConfig](aclmdlLoadWithConfig.md)接口指定模型加载时需要的配置信息，并进行模型加载。
5.  模型加载成功后，调用[aclmdlDestroyConfigHandle](aclmdlDestroyConfigHandle.md)接口销毁。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclmdlSetExternalWeightAddress(aclmdlConfigHandle *handle, const char *weightFileName, void *devPtr, size_t size)
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
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1164215556716"><a name="p1164215556716"></a><a name="p1164215556716"></a>handle</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1364212551074"><a name="p1364212551074"></a><a name="p1364212551074"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1264216557714"><a name="p1264216557714"></a><a name="p1264216557714"></a>模型加载的配置对象的指针。需提前调用<a href="aclmdlCreateConfigHandle.md">aclmdlCreateConfigHandle</a>接口创建该对象。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1264211554711"><a name="p1264211554711"></a><a name="p1264211554711"></a>weightFileName</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p106423556713"><a name="p106423556713"></a><a name="p106423556713"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p35961515104616"><a name="p35961515104616"></a><a name="p35961515104616"></a>外置权重文件名，不含路径。</p>
<p id="p1762216010545"><a name="p1762216010545"></a><a name="p1762216010545"></a>一般对om模型文件大小有限制或模型文件加密的场景下，需单独加载权重文件，因此需在构建模型时，将权重保存在单独的文件中。例如在使用ATC工具生成om文件时，将--external_weight参数设置为1（1表示将原始网络中的Const/Constant节点的权重保存在单独的文件中）。</p>
</td>
</tr>
<tr id="row1278904017717"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p664218551071"><a name="p664218551071"></a><a name="p664218551071"></a>devPtr</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1642155512710"><a name="p1642155512710"></a><a name="p1642155512710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p2030122311489"><a name="p2030122311489"></a><a name="p2030122311489"></a>Device上存放外置权重的内存地址指针。</p>
<p id="p750619584472"><a name="p750619584472"></a><a name="p750619584472"></a>该Device内存由用户自行管理，模型执行过程中不能释放该内存。</p>
</td>
</tr>
<tr id="row174374320716"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p164316555719"><a name="p164316555719"></a><a name="p164316555719"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p126436551477"><a name="p126436551477"></a><a name="p126436551477"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1864375512717"><a name="p1864375512717"></a><a name="p1864375512717"></a>内存的大小，单位Byte，需32字节对齐。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

