# aclmdlGetOutputDims<a name="ZH-CN_TOPIC_0000001312481581"></a>

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

根据模型描述信息获取指定的模型输出tensor的维度信息。

固定Shape场景下，通过该接口获取指定的模型输出tensor的维度信息。

动态Shape（动态Batch或动态分辨率）场景下，通过该接口获取最大档的维度信息。

## 函数原型<a name="section18474205015436"></a>

```
aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
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
<tbody><tr id="row134122191416"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p14132111415"><a name="p14132111415"></a><a name="p14132111415"></a>modelDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p144122141415"><a name="p144122141415"></a><a name="p144122141415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4414215143"><a name="p4414215143"></a><a name="p4414215143"></a>aclmdlDesc类型的指针。</p>
<p id="p35754134454"><a name="p35754134454"></a><a name="p35754134454"></a>需提前调用<a href="aclmdlCreateDesc.md">aclmdlCreateDesc</a>接口创建aclmdlDesc类型的数据。</p>
</td>
</tr>
<tr id="row218712903311"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p0984193455911"><a name="p0984193455911"></a><a name="p0984193455911"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p17984134125911"><a name="p17984134125911"></a><a name="p17984134125911"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p189841134135911"><a name="p189841134135911"></a><a name="p189841134135911"></a>指定获取第几个输出的Dims，index值从0开始。</p>
</td>
</tr>
<tr id="row118328235595"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p204083318531"><a name="p204083318531"></a><a name="p204083318531"></a>dims</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p184012339539"><a name="p184012339539"></a><a name="p184012339539"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p112981040205720"><a name="p112981040205720"></a><a name="p112981040205720"></a>输出维度信息的指针。</p>
<p id="p149381612192019"><a name="p149381612192019"></a><a name="p149381612192019"></a>若tensor的name长度大于127，则在输出的dims.name时，系统会将tensor的name转换为“acl_modelId_<em id="i678714213346"><a name="i678714213346"></a><a name="i678714213346"></a>${id}</em>_output_<em id="i4983860345"><a name="i4983860345"></a><a name="i4983860345"></a>${index}</em> _<em id="i12443055216"><a name="i12443055216"></a><a name="i12443055216"></a>${随机字符串}</em>   ”格式（如果转换后的tensor name与模型中已有的tensor name冲突，则会在转换后的name尾部增加“_<em id="i1388175614276"><a name="i1388175614276"></a><a name="i1388175614276"></a>${随机字符串}</em> ”，否则不会增加随机字符串），并在转换后的name与原name之间建立映射关系，用户可调用<a href="aclmdlGetTensorRealName.md">aclmdlGetTensorRealName</a>接口，传入转换后的name，获取原name（若向接口传入原name，则获取的还是原name）；若tensor的name长度小于或等于127，则在输出的dims.name时，按tensor的name输出。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section184151120582"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

