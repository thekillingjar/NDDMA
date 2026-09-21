# aclmdlSetAIPPPaddingParams<a name="ZH-CN_TOPIC_0000001265081350"></a>

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

动态AIPP场景下，设置补边相关的参数。

## 函数原型<a name="section2067518173415"></a>

```
aclError aclmdlSetAIPPPaddingParams(aclmdlAIPP *aippParmsSet, int8_t paddingSwitch,
int32_t paddingSizeTop, int32_t paddingSizeBottom,
int32_t paddingSizeLeft, int32_t paddingSizeRight,
uint64_t batchIndex)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>aippParmsSet</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p260514111135"><a name="p260514111135"></a><a name="p260514111135"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p956653619549"><a name="p956653619549"></a><a name="p956653619549"></a>动态AIPP参数对象的指针。</p>
<p id="p19724104413549"><a name="p19724104413549"></a><a name="p19724104413549"></a>需提前调用<a href="aclmdlCreateAIPP.md">aclmdlCreateAIPP</a>接口创建aclmdlAIPP类型的数据。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p371635485016"><a name="p371635485016"></a><a name="p371635485016"></a>paddingSwitch</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p69341916814"><a name="p69341916814"></a><a name="p69341916814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p19215102754518"><a name="p19215102754518"></a><a name="p19215102754518"></a>是否对图片执行补边操作，取值范围：</p>
<a name="ul11639192834513"></a><a name="ul11639192834513"></a><ul id="ul11639192834513"><li>0：不执行补边操作，设置为0时，则设置paddingSizeTop、paddingSizeBottom、paddingSizeLeft、paddingSizeRight参数无效</li><li>1：执行补边操作</li></ul>
<p id="p8480174054513"><a name="p8480174054513"></a><a name="p8480174054513"></a></p>
</td>
</tr>
<tr id="row1919192774810"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p27151254145011"><a name="p27151254145011"></a><a name="p27151254145011"></a>paddingSizeTop</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1092017278485"><a name="p1092017278485"></a><a name="p1092017278485"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p9164474"><a name="p9164474"></a><a name="p9164474"></a>在图片上方填充的值。</p>
<p id="p945392117488"><a name="p945392117488"></a><a name="p945392117488"></a>取值范围：[0, 32]</p>
</td>
</tr>
<tr id="row273214316518"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p173213436513"><a name="p173213436513"></a><a name="p173213436513"></a>paddingSizeBottom</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p57322438511"><a name="p57322438511"></a><a name="p57322438511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p25536001"><a name="p25536001"></a><a name="p25536001"></a>在图片下方填充的值。</p>
<p id="p11599623104811"><a name="p11599623104811"></a><a name="p11599623104811"></a>取值范围：[0, 32]</p>
</td>
</tr>
<tr id="row10945045185116"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p6945104565113"><a name="p6945104565113"></a><a name="p6945104565113"></a>paddingSizeLeft</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p694624595114"><a name="p694624595114"></a><a name="p694624595114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p16999516"><a name="p16999516"></a><a name="p16999516"></a>在图片左方填充的值。</p>
<p id="p177651925114816"><a name="p177651925114816"></a><a name="p177651925114816"></a>取值范围：[0, 32]</p>
</td>
</tr>
<tr id="row2197115675116"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p51978563519"><a name="p51978563519"></a><a name="p51978563519"></a>paddingSizeRight</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p111973565519"><a name="p111973565519"></a><a name="p111973565519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p3773844"><a name="p3773844"></a><a name="p3773844"></a>在图片右方填充的值。</p>
<p id="p1780132716487"><a name="p1780132716487"></a><a name="p1780132716487"></a>取值范围：[0, 32]</p>
</td>
</tr>
<tr id="row67101958125112"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p11710195817517"><a name="p11710195817517"></a><a name="p11710195817517"></a>batchIndex</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1871015811517"><a name="p1871015811517"></a><a name="p1871015811517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p157101358155119"><a name="p157101358155119"></a><a name="p157101358155119"></a>指定对第几个Batch上的图片执行补边操作。</p>
<p id="p370612296453"><a name="p370612296453"></a><a name="p370612296453"></a>取值范围：[0,batchSize)</p>
<p id="p1211191404610"><a name="p1211191404610"></a><a name="p1211191404610"></a>batchSize是在调用<a href="aclmdlCreateAIPP.md">aclmdlCreateAIPP</a>接口创建aclmdlAIPP类型的数据时设置。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section15770391345"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section1793822017455"></a>

补边之后，图片的宽必须小于等于1080。

