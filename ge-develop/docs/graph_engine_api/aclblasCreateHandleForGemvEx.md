# aclblasCreateHandleForGemvEx<a name="ZH-CN_TOPIC_0000001265400270"></a>

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

## 功能说明<a name="section36583473819"></a>

创建矩阵-向量乘的handle，输入数据、输出数据的数据类型通过入参设置。创建handle成功后，需调用[aclopExecWithHandle](aclopExecWithHandle.md)接口执行算子。

A、x、y的数据类型仅支持以下组合：

<a name="table159219490272"></a>
<table><thead align="left"><tr id="row492104962720"><th class="cellrowborder" valign="top" width="29.03%" id="mcps1.1.4.1.1"><p id="p59254916272"><a name="p59254916272"></a><a name="p59254916272"></a>A的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="24.529999999999998%" id="mcps1.1.4.1.2"><p id="p59254962710"><a name="p59254962710"></a><a name="p59254962710"></a>x的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="46.44%" id="mcps1.1.4.1.3"><p id="p6938490278"><a name="p6938490278"></a><a name="p6938490278"></a>y的数据类型</p>
</th>
</tr>
</thead>
<tbody><tr id="row189354917275"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p4395121611592"><a name="p4395121611592"></a><a name="p4395121611592"></a><span id="ph7534113612588"><a name="ph7534113612588"></a><a name="ph7534113612588"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p1739611160597"><a name="p1739611160597"></a><a name="p1739611160597"></a><span id="ph38261649205"><a name="ph38261649205"></a><a name="ph38261649205"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p16884173220479"><a name="p16884173220479"></a><a name="p16884173220479"></a><span id="ph29122534011"><a name="ph29122534011"></a><a name="ph29122534011"></a>aclFloat16</span></p>
</td>
</tr>
<tr id="row53965163597"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p1739611611598"><a name="p1739611611598"></a><a name="p1739611611598"></a><span id="ph13111861803"><a name="ph13111861803"></a><a name="ph13111861803"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p202273233281"><a name="p202273233281"></a><a name="p202273233281"></a><span id="ph138370511806"><a name="ph138370511806"></a><a name="ph138370511806"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p12884232124718"><a name="p12884232124718"></a><a name="p12884232124718"></a>float(float32)</p>
</td>
</tr>
<tr id="row739610165599"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p5833183416288"><a name="p5833183416288"></a><a name="p5833183416288"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p739613163596"><a name="p739613163596"></a><a name="p739613163596"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p361725213462"><a name="p361725213462"></a><a name="p361725213462"></a>float(float32)</p>
</td>
</tr>
<tr id="row2073117269289"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p47320263289"><a name="p47320263289"></a><a name="p47320263289"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p77321926142816"><a name="p77321926142816"></a><a name="p77321926142816"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p14732132620282"><a name="p14732132620282"></a><a name="p14732132620282"></a>int32_t</p>
</td>
</tr>
</tbody>
</table>

## 函数原型<a name="section13230182415108"></a>

```
aclError aclblasCreateHandleForGemvEx(aclTransType transA,
int m,
int n,
aclDataType dataTypeA,
aclDataType dataTypeX,
aclDataType dataTypeY,
aclComputeType type,
aclopHandle **handle)
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
<tbody><tr id="row1340613017456"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p3406173064514"><a name="p3406173064514"></a><a name="p3406173064514"></a>transA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p86916447151"><a name="p86916447151"></a><a name="p86916447151"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1169144416159"><a name="p1169144416159"></a><a name="p1169144416159"></a>A矩阵是否转置的标记。</p>
</td>
</tr>
<tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p14581121011102"><a name="p14581121011102"></a><a name="p14581121011102"></a>m</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p59813581663"><a name="p59813581663"></a><a name="p59813581663"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p129801158560"><a name="p129801158560"></a><a name="p129801158560"></a>矩阵A的行数，存储矩阵乘数据时，行优先。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p858113104101"><a name="p858113104101"></a><a name="p858113104101"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1897918581669"><a name="p1897918581669"></a><a name="p1897918581669"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10978205810619"><a name="p10978205810619"></a><a name="p10978205810619"></a>矩阵A的列数。</p>
</td>
</tr>
<tr id="row17409124312112"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p153511441434"><a name="p153511441434"></a><a name="p153511441434"></a>dataTypeA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1735112415320"><a name="p1735112415320"></a><a name="p1735112415320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p93511041338"><a name="p93511041338"></a><a name="p93511041338"></a>矩阵A的数据类型。</p>
</td>
</tr>
<tr id="row05461045211"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p520119381336"><a name="p520119381336"></a><a name="p520119381336"></a>dataTypeX</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p18201183816315"><a name="p18201183816315"></a><a name="p18201183816315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p94642011153217"><a name="p94642011153217"></a><a name="p94642011153217"></a>向量x的数据类型。</p>
</td>
</tr>
<tr id="row164314237210"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p6451154839"><a name="p6451154839"></a><a name="p6451154839"></a>dataTypeY</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p6453541630"><a name="p6453541630"></a><a name="p6453541630"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p345154531"><a name="p345154531"></a><a name="p345154531"></a>向量y的数据类型。</p>
</td>
</tr>
<tr id="row1534184820455"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1234148184510"><a name="p1234148184510"></a><a name="p1234148184510"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p134970011178"><a name="p134970011178"></a><a name="p134970011178"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p174976021713"><a name="p174976021713"></a><a name="p174976021713"></a>计算精度，默认高精度。</p>
</td>
</tr>
<tr id="row344817319212"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p155821010201017"><a name="p155821010201017"></a><a name="p155821010201017"></a>handle</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p196675814611"><a name="p196675814611"></a><a name="p196675814611"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1369101110474"><a name="p1369101110474"></a><a name="p1369101110474"></a>“算子执行handle的指针”的指针。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

