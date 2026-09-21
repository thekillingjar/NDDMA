# aclblasGemvEx<a name="ZH-CN_TOPIC_0000001265241730"></a>

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

## 功能说明<a name="section8698174414302"></a>

执行矩阵-向量的乘法，y = αAx + βy，输入数据、输出数据的数据类型通过入参设置。异步接口。

A、x、y的数据类型仅支持以下组合， α和β的数据类型与y一致。

<a name="table159219490272"></a>
<table><thead align="left"><tr id="row492104962720"><th class="cellrowborder" valign="top" width="29.03%" id="mcps1.1.4.1.1"><p id="p59254916272"><a name="p59254916272"></a><a name="p59254916272"></a>A的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="24.529999999999998%" id="mcps1.1.4.1.2"><p id="p59254962710"><a name="p59254962710"></a><a name="p59254962710"></a>x的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="46.44%" id="mcps1.1.4.1.3"><p id="p6938490278"><a name="p6938490278"></a><a name="p6938490278"></a>y的数据类型</p>
</th>
</tr>
</thead>
<tbody><tr id="row189354917275"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p14581121011102"><a name="p14581121011102"></a><a name="p14581121011102"></a><span id="ph7534113612588"><a name="ph7534113612588"></a><a name="ph7534113612588"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p59813581663"><a name="p59813581663"></a><a name="p59813581663"></a><span id="ph10957184218597"><a name="ph10957184218597"></a><a name="ph10957184218597"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p16884173220479"><a name="p16884173220479"></a><a name="p16884173220479"></a><span id="ph125814712594"><a name="ph125814712594"></a><a name="ph125814712594"></a>aclFloat16</span></p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p858113104101"><a name="p858113104101"></a><a name="p858113104101"></a><span id="ph2862340195912"><a name="ph2862340195912"></a><a name="ph2862340195912"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p202273233281"><a name="p202273233281"></a><a name="p202273233281"></a><span id="ph6919164495917"><a name="ph6919164495917"></a><a name="ph6919164495917"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p12884232124718"><a name="p12884232124718"></a><a name="p12884232124718"></a>float(float32)</p>
</td>
</tr>
<tr id="row344817319212"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p5833183416288"><a name="p5833183416288"></a><a name="p5833183416288"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p196675814611"><a name="p196675814611"></a><a name="p196675814611"></a>int8_t</p>
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

## 函数原型<a name="section7496145153016"></a>

```
aclError aclblasGemvEx(aclTransType transA,
int m,
int n,
const void *alpha,
const void *a,
int lda,
aclDataType dataTypeA,
const void *x,
int incx,
aclDataType dataTypeX,
const void *beta,
void *y,
int incy,
aclDataType dataTypeY,
aclComputeType type,
aclrtStream stream)
```

## 参数说明<a name="section1492335418306"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row0694442151"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1269164421511"><a name="p1269164421511"></a><a name="p1269164421511"></a>transA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p86916447151"><a name="p86916447151"></a><a name="p86916447151"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1169144416159"><a name="p1169144416159"></a><a name="p1169144416159"></a>A矩阵是否转置的标记。</p>
</td>
</tr>
<tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p923414586318"><a name="p923414586318"></a><a name="p923414586318"></a>m</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p22333584316"><a name="p22333584316"></a><a name="p22333584316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p129801158560"><a name="p129801158560"></a><a name="p129801158560"></a>矩阵A的行数，存储矩阵乘数据时，行优先。</p>
</td>
</tr>
<tr id="row745163011317"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1545220301039"><a name="p1545220301039"></a><a name="p1545220301039"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p445213305311"><a name="p445213305311"></a><a name="p445213305311"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10978205810619"><a name="p10978205810619"></a><a name="p10978205810619"></a>矩阵A的列数。</p>
</td>
</tr>
<tr id="row79830322035"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p59833321038"><a name="p59833321038"></a><a name="p59833321038"></a>alpha</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p7983143212316"><a name="p7983143212316"></a><a name="p7983143212316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p49774585613"><a name="p49774585613"></a><a name="p49774585613"></a>用于执行乘操作的标量α的指针。</p>
</td>
</tr>
<tr id="row108515351035"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p168612352316"><a name="p168612352316"></a><a name="p168612352316"></a>a</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p88663510319"><a name="p88663510319"></a><a name="p88663510319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p88623513312"><a name="p88623513312"></a><a name="p88623513312"></a>矩阵A的指针。</p>
</td>
</tr>
<tr id="row17902160121619"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p3902808164"><a name="p3902808164"></a><a name="p3902808164"></a>lda</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p119027016167"><a name="p119027016167"></a><a name="p119027016167"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p690220161611"><a name="p690220161611"></a><a name="p690220161611"></a>A矩阵的主维，此时选择转置，按行优先，则lda为A的列数。预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row1635044115317"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p153511441434"><a name="p153511441434"></a><a name="p153511441434"></a>dataTypeA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1735112415320"><a name="p1735112415320"></a><a name="p1735112415320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p93511041338"><a name="p93511041338"></a><a name="p93511041338"></a>矩阵A的数据类型。</p>
</td>
</tr>
<tr id="row3823194314315"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p13824194316314"><a name="p13824194316314"></a><a name="p13824194316314"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1982454315317"><a name="p1982454315317"></a><a name="p1982454315317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p139731758764"><a name="p139731758764"></a><a name="p139731758764"></a>向量x的指针。</p>
</td>
</tr>
<tr id="row108251333171611"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p3825123315162"><a name="p3825123315162"></a><a name="p3825123315162"></a>incx</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p48251533191613"><a name="p48251533191613"></a><a name="p48251533191613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p38257337169"><a name="p38257337169"></a><a name="p38257337169"></a>x连续元素之间的步长。</p>
<p id="p980514144712"><a name="p980514144712"></a><a name="p980514144712"></a>预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row1620119387311"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p520119381336"><a name="p520119381336"></a><a name="p520119381336"></a>dataTypeX</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p18201183816315"><a name="p18201183816315"></a><a name="p18201183816315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p189705585618"><a name="p189705585618"></a><a name="p189705585618"></a>向量x的数据类型。</p>
</td>
</tr>
<tr id="row16553471134"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p166551147135"><a name="p166551147135"></a><a name="p166551147135"></a>beta</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p76554477317"><a name="p76554477317"></a><a name="p76554477317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p465511471933"><a name="p465511471933"></a><a name="p465511471933"></a>用于执行乘操作的标量β的指针。</p>
</td>
</tr>
<tr id="row163443501631"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p4345195015312"><a name="p4345195015312"></a><a name="p4345195015312"></a>y</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1234514507318"><a name="p1234514507318"></a><a name="p1234514507318"></a>输入&amp;输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p23452501136"><a name="p23452501136"></a><a name="p23452501136"></a>向量y的指针。</p>
</td>
</tr>
<tr id="row164744781611"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1647547191620"><a name="p1647547191620"></a><a name="p1647547191620"></a>incy</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p20647154781616"><a name="p20647154781616"></a><a name="p20647154781616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1364724712167"><a name="p1364724712167"></a><a name="p1364724712167"></a>y连续元素之间的步长。</p>
<p id="p20982416124712"><a name="p20982416124712"></a><a name="p20982416124712"></a>预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row145854238"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p6451154839"><a name="p6451154839"></a><a name="p6451154839"></a>dataTypeY</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p6453541630"><a name="p6453541630"></a><a name="p6453541630"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p345154531"><a name="p345154531"></a><a name="p345154531"></a>向量y的数据类型。</p>
</td>
</tr>
<tr id="row84971506178"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p749715061712"><a name="p749715061712"></a><a name="p749715061712"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p134970011178"><a name="p134970011178"></a><a name="p134970011178"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p174976021713"><a name="p174976021713"></a><a name="p174976021713"></a>计算精度，默认高精度。</p>
</td>
</tr>
<tr id="row11522151110616"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p105222111862"><a name="p105222111862"></a><a name="p105222111862"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p152214112616"><a name="p152214112616"></a><a name="p152214112616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p052281118620"><a name="p052281118620"></a><a name="p052281118620"></a>执行算子所在的Stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section59071758153012"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

