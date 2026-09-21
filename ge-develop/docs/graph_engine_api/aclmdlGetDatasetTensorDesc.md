# aclmdlGetDatasetTensorDesc<a name="ZH-CN_TOPIC_0000001312721709"></a>

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

## 功能说明<a name="section0117164520431"></a>

如果模型输入或输出的Shape是动态的，模型在模型执行之后可以调用本接口从aclmdlDataset类型的数据（该类型用于描述模型推理时的输入数据、输出数据）中获取指定的输入或者输出的Tensor描述信息。

**典型场景举例**：如果模型输入Shape是动态的，在模型执行之前调[aclmdlSetDatasetTensorDesc](aclmdlSetDatasetTensorDesc.md)设置该输入的tensor描述信息，在模型执行之后，调用aclmdlGetDatasetTensorDesc接口获取模型动态输出的Tensor描述信息。

## 函数原型<a name="section133154911438"></a>

```
aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index)
```

## 参数说明<a name="section13616184164416"></a>

<a name="table133342381612"></a>
<table><thead align="left"><tr id="row173342038467"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p1633411386618"><a name="p1633411386618"></a><a name="p1633411386618"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p8334163811620"><a name="p8334163811620"></a><a name="p8334163811620"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p03354381462"><a name="p03354381462"></a><a name="p03354381462"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row12335193818617"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p73353381668"><a name="p73353381668"></a><a name="p73353381668"></a>dataset</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p8335203817611"><a name="p8335203817611"></a><a name="p8335203817611"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10329145511613"><a name="p10329145511613"></a><a name="p10329145511613"></a>模型执行的输入或输出数据指针。</p>
</td>
</tr>
<tr id="row197051230493"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p147061823144910"><a name="p147061823144910"></a><a name="p147061823144910"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p16706823114915"><a name="p16706823114915"></a><a name="p16706823114915"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p137067239495"><a name="p137067239495"></a><a name="p137067239495"></a>表示第几个输入或输出的序号。</p>
<p id="p16627181015251"><a name="p16627181015251"></a><a name="p16627181015251"></a>模型存在多个输入、输出时，为避免序号出错，可以先调用<a href="aclmdlGetInputNameByIndex.md">aclmdlGetInputNameByIndex</a>、<a href="aclmdlGetOutputNameByIndex.md">aclmdlGetOutputNameByIndex</a>接口获取输入、输出的名称，根据输入、输出名称所对应的index来设置。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section162895447"></a>

返回指定输入或输出的tensor描述信息。

