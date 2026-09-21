# aclmdlSetDynamicHWSize<a name="ZH-CN_TOPIC_0000001265241674"></a>

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

## 功能说明<a name="section166231326115013"></a>

动态分辨率场景下，在模型执行前调用本接口设置模型推理时输入图片的高和宽。

## 函数原型<a name="section244219302509"></a>

```
aclError aclmdlSetDynamicHWSize(uint32_t modelId, aclmdlDataset *dataset, size_t index, uint64_t height, uint64_t width)
```

## 参数说明<a name="section163301034105012"></a>

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
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p8997624142310"><a name="p8997624142310"></a><a name="p8997624142310"></a>模型ID。</p>
<p id="p57291851112517"><a name="p57291851112517"></a><a name="p57291851112517"></a>调用<a href="aclmdlLoadFromFile.md">aclmdlLoadFromFile</a>接口/<a href="aclmdlLoadFromMem.md">aclmdlLoadFromMem</a>接口/<a href="aclmdlLoadFromFileWithMem.md">aclmdlLoadFromFileWithMem</a>接口/<a href="aclmdlLoadFromMemWithMem.md">aclmdlLoadFromMemWithMem</a>接口加载模型成功后，会返回模型ID。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p19341911481"><a name="p19341911481"></a><a name="p19341911481"></a>dataset</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p69341916814"><a name="p69341916814"></a><a name="p69341916814"></a>输入&amp;输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p93532063519"><a name="p93532063519"></a><a name="p93532063519"></a>模型推理的输入数据的指针。</p>
<p id="p4388859358"><a name="p4388859358"></a><a name="p4388859358"></a>使用aclmdlDataset类型的数据描述模型推理时的输入数据，输入的内存地址、内存大小用<span id="ph44438521387"><a name="ph44438521387"></a><a name="ph44438521387"></a>aclDataBuffer</span>类型的数据来描述。</p>
</td>
</tr>
<tr id="row147732496171"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p072371917818"><a name="p072371917818"></a><a name="p072371917818"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p177231319387"><a name="p177231319387"></a><a name="p177231319387"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p64351912165911"><a name="p64351912165911"></a><a name="p64351912165911"></a>标识动态分辨率输入的输入index，需调用<a href="aclmdlGetInputIndexByName.md">aclmdlGetInputIndexByName</a>接口获取，输入名称固定为ACL_DYNAMIC_TENSOR_NAME。</p>
</td>
</tr>
<tr id="row955943603613"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p555311348580"><a name="p555311348580"></a><a name="p555311348580"></a>height</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p12554153418587"><a name="p12554153418587"></a><a name="p12554153418587"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p15554173414586"><a name="p15554173414586"></a><a name="p15554173414586"></a>需设置的H值。</p>
<p id="p1871113225319"><a name="p1871113225319"></a><a name="p1871113225319"></a>此处设置的分辨率只能是模型构建时设置的分辨率档位中的某一档。</p>
<p id="p69986413380"><a name="p69986413380"></a><a name="p69986413380"></a>如果不清楚模型构建时的分辨率档位，也可以调用<a href="aclmdlGetDynamicHW.md">aclmdlGetDynamicHW</a>接口获取指定模型支持的分辨率档位数以及每一档中的宽、高。</p>
</td>
</tr>
<tr id="row2604183119586"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p12605123125813"><a name="p12605123125813"></a><a name="p12605123125813"></a>width</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p18605153110585"><a name="p18605153110585"></a><a name="p18605153110585"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p176057315583"><a name="p176057315583"></a><a name="p176057315583"></a>需设置的W值。</p>
<p id="p033419316355"><a name="p033419316355"></a><a name="p033419316355"></a>此处设置的分辨率只能是模型构建时设置的分辨率档位中的某一档。</p>
<p id="p10334163103517"><a name="p10334163103517"></a><a name="p10334163103517"></a>如果不清楚模型构建时的分辨率档位，也可以调用<a href="aclmdlGetDynamicHW.md">aclmdlGetDynamicHW</a>接口获取指定模型支持的分辨率档位数以及每一档中的宽、高。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section7216174405014"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

