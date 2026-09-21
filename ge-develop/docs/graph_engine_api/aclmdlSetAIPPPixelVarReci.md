# aclmdlSetAIPPPixelVarReci<a name="ZH-CN_TOPIC_0000001265241782"></a>

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

## 功能说明<a name="section259105813316"></a>

动态AIPP场景下，设置通道的方差。

## 函数原型<a name="section2067518173415"></a>

```
aclError aclmdlSetAIPPPixelVarReci(aclmdlAIPP *aippParmsSet,
float dtcPixelVarReciChn0,
float dtcPixelVarReciChn1,
float dtcPixelVarReciChn2,
float dtcPixelVarReciChn3,
uint64_t batchIndex)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18.55%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.450000000000001%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18.55%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>aippParmsSet</p>
</td>
<td class="cellrowborder" valign="top" width="13.450000000000001%" headers="mcps1.1.4.1.2 "><p id="p260514111135"><a name="p260514111135"></a><a name="p260514111135"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p956653619549"><a name="p956653619549"></a><a name="p956653619549"></a>动态AIPP参数对象的指针。</p>
<p id="p19724104413549"><a name="p19724104413549"></a><a name="p19724104413549"></a>需提前调用<a href="aclmdlCreateAIPP.md">aclmdlCreateAIPP</a>接口创建aclmdlAIPP类型的数据。</p>
</td>
</tr>
<tr id="row1919192774810"><td class="cellrowborder" valign="top" width="18.55%" headers="mcps1.1.4.1.1 "><p id="p27151254145011"><a name="p27151254145011"></a><a name="p27151254145011"></a>dtcPixelVarReciChn0</p>
</td>
<td class="cellrowborder" valign="top" width="13.450000000000001%" headers="mcps1.1.4.1.2 "><p id="p1092017278485"><a name="p1092017278485"></a><a name="p1092017278485"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p67055397"><a name="p67055397"></a><a name="p67055397"></a>通道0的方差的倒数，默认值为1.0。</p>
<p id="p45548209527"><a name="p45548209527"></a><a name="p45548209527"></a>取值范围：[-65504, 65504]</p>
</td>
</tr>
<tr id="row273214316518"><td class="cellrowborder" valign="top" width="18.55%" headers="mcps1.1.4.1.1 "><p id="p173213436513"><a name="p173213436513"></a><a name="p173213436513"></a>dtcPixelVarReciChn1</p>
</td>
<td class="cellrowborder" valign="top" width="13.450000000000001%" headers="mcps1.1.4.1.2 "><p id="p57322438511"><a name="p57322438511"></a><a name="p57322438511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p26193631"><a name="p26193631"></a><a name="p26193631"></a>通道1的方差的倒数，默认值为1.0。</p>
<p id="p271542219529"><a name="p271542219529"></a><a name="p271542219529"></a>取值范围：[-65504, 65504]</p>
</td>
</tr>
<tr id="row10945045185116"><td class="cellrowborder" valign="top" width="18.55%" headers="mcps1.1.4.1.1 "><p id="p6945104565113"><a name="p6945104565113"></a><a name="p6945104565113"></a>dtcPixelVarReciChn2</p>
</td>
<td class="cellrowborder" valign="top" width="13.450000000000001%" headers="mcps1.1.4.1.2 "><p id="p694624595114"><a name="p694624595114"></a><a name="p694624595114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p6517326"><a name="p6517326"></a><a name="p6517326"></a>通道2的方差的倒数，默认值为1.0。</p>
<p id="p17811162614529"><a name="p17811162614529"></a><a name="p17811162614529"></a>取值范围：[-65504, 65504]</p>
</td>
</tr>
<tr id="row2197115675116"><td class="cellrowborder" valign="top" width="18.55%" headers="mcps1.1.4.1.1 "><p id="p51978563519"><a name="p51978563519"></a><a name="p51978563519"></a>dtcPixelVarReciChn3</p>
</td>
<td class="cellrowborder" valign="top" width="13.450000000000001%" headers="mcps1.1.4.1.2 "><p id="p111973565519"><a name="p111973565519"></a><a name="p111973565519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p13462075"><a name="p13462075"></a><a name="p13462075"></a>通道3的方差的倒数，默认值为1.0。如果只有3个通道，则将该参数设置为1.0。</p>
<p id="p2802192955210"><a name="p2802192955210"></a><a name="p2802192955210"></a>取值范围：[-65504, 65504]</p>
</td>
</tr>
<tr id="row67101958125112"><td class="cellrowborder" valign="top" width="18.55%" headers="mcps1.1.4.1.1 "><p id="p11710195817517"><a name="p11710195817517"></a><a name="p11710195817517"></a>batchIndex</p>
</td>
<td class="cellrowborder" valign="top" width="13.450000000000001%" headers="mcps1.1.4.1.2 "><p id="p1871015811517"><a name="p1871015811517"></a><a name="p1871015811517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p157101358155119"><a name="p157101358155119"></a><a name="p157101358155119"></a>指定对第几个Batch上的图片设置通道的方差。</p>
<p id="p370612296453"><a name="p370612296453"></a><a name="p370612296453"></a>取值范围：[0,batchSize)</p>
<p id="p1211191404610"><a name="p1211191404610"></a><a name="p1211191404610"></a>batchSize是在调用<a href="aclmdlCreateAIPP.md">aclmdlCreateAIPP</a>接口创建aclmdlAIPP类型的数据时设置。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section15770391345"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

