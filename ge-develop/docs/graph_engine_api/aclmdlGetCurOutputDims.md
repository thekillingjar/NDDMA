# aclmdlGetCurOutputDims<a name="ZH-CN_TOPIC_0000001265400378"></a>

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

## 功能说明<a name="section187304428437"></a>

根据模型描述信息获取指定的模型输出tensor的实际维度信息。

## 函数原型<a name="section18474205015436"></a>

```
aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
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
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p177891510145214"><a name="p177891510145214"></a><a name="p177891510145214"></a>输出实际维度信息的指针。</p>
<p id="p846623563418"><a name="p846623563418"></a><a name="p846623563418"></a>若tensor的name长度大于127，则在输出的dims.name时，系统会将tensor的name转换为“acl_modelId_<em id="zh-cn_topic_0000001312481581_i678714213346"><a name="zh-cn_topic_0000001312481581_i678714213346"></a><a name="zh-cn_topic_0000001312481581_i678714213346"></a>${id}</em>_output_<em id="zh-cn_topic_0000001312481581_i4983860345"><a name="zh-cn_topic_0000001312481581_i4983860345"></a><a name="zh-cn_topic_0000001312481581_i4983860345"></a>${index}</em> _<em id="zh-cn_topic_0000001312481581_i12443055216"><a name="zh-cn_topic_0000001312481581_i12443055216"></a><a name="zh-cn_topic_0000001312481581_i12443055216"></a>${随机字符串}</em>   ”格式（如果转换后的tensor name与模型中已有的tensor name冲突，则会在转换后的name尾部增加“_<em id="zh-cn_topic_0000001312481581_i1388175614276"><a name="zh-cn_topic_0000001312481581_i1388175614276"></a><a name="zh-cn_topic_0000001312481581_i1388175614276"></a>${随机字符串}</em> ”，否则不会增加随机字符串），并在转换后的name与原name之间建立映射关系，用户可调用<a href="aclmdlGetTensorRealName.md">aclmdlGetTensorRealName</a>接口，传入转换后的name，获取原name（若向接口传入原name，则获取的还是原name）；若tensor的name长度小于或等于127，则在输出的dims.name时，按tensor的name输出。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section184151120582"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section153995348166"></a>

当前仅支持通过本接口获取以下场景中的模型输出tensor的维度信息：

-   通过模型转换设置多档Batch size或分辨率或维度值，实现动态Batch或动态分辨率或动态维度（ND格式）时，如果用户已调用[aclmdlSetDynamicBatchSize](aclmdlSetDynamicBatchSize.md)设置Batch、或调用[aclmdlSetDynamicHWSize](aclmdlSetDynamicHWSize.md)接口设置输入图片的宽高、或调用[aclmdlSetInputDynamicDims](aclmdlSetInputDynamicDims.md)接口设置某动态维度的值，则可通过该接口获取指定模型输出tensor的实际维度信息；如果用户未调用[aclmdlSetDynamicBatchSize](aclmdlSetDynamicBatchSize.md)接口、或[aclmdlSetDynamicHWSize](aclmdlSetDynamicHWSize.md)接口、或[aclmdlSetInputDynamicDims](aclmdlSetInputDynamicDims.md)接口，则通过该接口可获取最大档的维度信息。
-   固定Shape场景下，通过该接口获取指定的模型输出tensor的维度信息。

