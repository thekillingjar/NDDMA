# aclmdlSetAIPPScfParams<a name="ZH-CN_TOPIC_0000001265400626"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>x</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

动态AIPP场景下，设置缩放相关的参数。

## 函数原型<a name="section2067518173415"></a>

```
aclError aclmdlSetAIPPScfParams(aclmdlAIPP *aippParmsSet, int8_t scfSwitch,
int32_t scfInputSizeW, int32_t scfInputSizeH,
int32_t scfOutputSizeW, int32_t scfOutputSizeH,
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
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p371635485016"><a name="p371635485016"></a><a name="p371635485016"></a>scfSwitch</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p69341916814"><a name="p69341916814"></a><a name="p69341916814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p0680161418442"><a name="p0680161418442"></a><a name="p0680161418442"></a>是否对图片执行缩放操作，取值范围：</p>
<a name="ul5249516124415"></a><a name="ul5249516124415"></a><ul id="ul5249516124415"><li>0：不执行缩放操作，设置为0时，则设置scfInputSizeW、scfInputSizeH、scfOutputSizeW、scfOutputSizeH参数无效</li><li>1：执行缩放操作</li></ul>
<p id="p73761526114412"><a name="p73761526114412"></a><a name="p73761526114412"></a></p>
</td>
</tr>
<tr id="row1919192774810"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p27151254145011"><a name="p27151254145011"></a><a name="p27151254145011"></a>scfInputSizeW</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1092017278485"><a name="p1092017278485"></a><a name="p1092017278485"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p53371562501"><a name="p53371562501"></a><a name="p53371562501"></a>缩放前图片的宽。</p>
<p id="p439391114517"><a name="p439391114517"></a><a name="p439391114517"></a>取值范围：[16,4096]</p>
<p id="p1661112814329"><a name="p1661112814329"></a><a name="p1661112814329"></a>若开启了抠图功能，则缩放前图片的宽与<a href="aclmdlSetAIPPCropParams.md">抠图区域的宽度</a>保持一致；若未开启抠图功能，则缩放前图片的宽与<a href="aclmdlSetAIPPSrcImageSize.md">原始图片的宽</a>保持一致。</p>
</td>
</tr>
<tr id="row273214316518"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p173213436513"><a name="p173213436513"></a><a name="p173213436513"></a>scfInputSizeH</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p57322438511"><a name="p57322438511"></a><a name="p57322438511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4732843135114"><a name="p4732843135114"></a><a name="p4732843135114"></a>缩放前图片的高。</p>
<p id="p746591324518"><a name="p746591324518"></a><a name="p746591324518"></a>取值范围：[16,4096]</p>
<p id="p12921132910340"><a name="p12921132910340"></a><a name="p12921132910340"></a>若开启了抠图功能，则缩放前图片的高与<a href="aclmdlSetAIPPCropParams.md">抠图区域的高度</a>保持一致；若未开启抠图功能，则缩放前图片的高与<a href="aclmdlSetAIPPSrcImageSize.md">原始图片的高</a>保持一致。</p>
</td>
</tr>
<tr id="row10945045185116"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p6945104565113"><a name="p6945104565113"></a><a name="p6945104565113"></a>scfOutputSizeW</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p694624595114"><a name="p694624595114"></a><a name="p694624595114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p17946845175119"><a name="p17946845175119"></a><a name="p17946845175119"></a>缩放后图片的宽。</p>
<p id="p8883151513459"><a name="p8883151513459"></a><a name="p8883151513459"></a>取值范围：[16,1920]</p>
</td>
</tr>
<tr id="row2197115675116"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p51978563519"><a name="p51978563519"></a><a name="p51978563519"></a>scfOutputSizeH</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p111973565519"><a name="p111973565519"></a><a name="p111973565519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p01975560516"><a name="p01975560516"></a><a name="p01975560516"></a>缩放后图片的高。</p>
<p id="p18625181713455"><a name="p18625181713455"></a><a name="p18625181713455"></a>取值范围：[16,4096]</p>
</td>
</tr>
<tr id="row67101958125112"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p11710195817517"><a name="p11710195817517"></a><a name="p11710195817517"></a>batchIndex</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1871015811517"><a name="p1871015811517"></a><a name="p1871015811517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p157101358155119"><a name="p157101358155119"></a><a name="p157101358155119"></a>指定对第几个Batch上的图片执行缩放操作。</p>
<p id="p370612296453"><a name="p370612296453"></a><a name="p370612296453"></a>取值范围：[0,batchSize)</p>
<p id="p1211191404610"><a name="p1211191404610"></a><a name="p1211191404610"></a>batchSize是在调用<a href="aclmdlCreateAIPP.md">aclmdlCreateAIPP</a>接口创建aclmdlAIPP类型的数据时设置。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section15770391345"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section112191137135115"></a>

缩放比例scfOutputSizeW/scfInputSizeW∈\[1/16,16\]、scfOutputSizeH/scfInputSizeH∈\[1/16,16\]。

