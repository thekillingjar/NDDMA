# aclblasGemmEx<a name="ZH-CN_TOPIC_0000001312481797"></a>

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

## 功能说明<a name="section1214123264814"></a>

执行矩阵-矩阵的乘法，C = αAB + βC，输入数据、输出数据的数据类型通过入参设置。异步接口。

A、B、C的数据类型仅支持以下组合， α和β的数据类型与C一致。

<a name="table159219490272"></a>
<table><thead align="left"><tr id="row492104962720"><th class="cellrowborder" valign="top" width="29.03%" id="mcps1.1.4.1.1"><p id="p59254916272"><a name="p59254916272"></a><a name="p59254916272"></a>A的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="24.529999999999998%" id="mcps1.1.4.1.2"><p id="p59254962710"><a name="p59254962710"></a><a name="p59254962710"></a>B的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="46.44%" id="mcps1.1.4.1.3"><p id="p6938490278"><a name="p6938490278"></a><a name="p6938490278"></a>C的数据类型</p>
</th>
</tr>
</thead>
<tbody><tr id="row189354917275"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p14581121011102"><a name="p14581121011102"></a><a name="p14581121011102"></a><span id="ph7534113612588"><a name="ph7534113612588"></a><a name="ph7534113612588"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p59813581663"><a name="p59813581663"></a><a name="p59813581663"></a><span id="ph115718311211"><a name="ph115718311211"></a><a name="ph115718311211"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p129801158560"><a name="p129801158560"></a><a name="p129801158560"></a><span id="ph33551271828"><a name="ph33551271828"></a><a name="ph33551271828"></a>aclFloat16</span></p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="29.03%" headers="mcps1.1.4.1.1 "><p id="p858113104101"><a name="p858113104101"></a><a name="p858113104101"></a><span id="ph20264329719"><a name="ph20264329719"></a><a name="ph20264329719"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="24.529999999999998%" headers="mcps1.1.4.1.2 "><p id="p202273233281"><a name="p202273233281"></a><a name="p202273233281"></a><span id="ph17132205022"><a name="ph17132205022"></a><a name="ph17132205022"></a>aclFloat16</span></p>
</td>
<td class="cellrowborder" valign="top" width="46.44%" headers="mcps1.1.4.1.3 "><p id="p10978205810619"><a name="p10978205810619"></a><a name="p10978205810619"></a>float(float32)</p>
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

## 函数原型<a name="section51287362487"></a>

```
aclError aclblasGemmEx(aclTransType transA,
aclTransType transB,
aclTransType transC,
int m,
int n,
int k,
const void *alpha,
const void *matrixA,
int lda,
aclDataType dataTypeA,
const void *matrixB,
int ldb,
aclDataType dataTypeB,
const void *beta,
void *matrixC,
int ldc,
aclDataType dataTypeC,
aclComputeType type,
aclrtStream stream)
```

## 参数说明<a name="section561319419480"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1343304919287"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p2043314915286"><a name="p2043314915286"></a><a name="p2043314915286"></a>transA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p943314495283"><a name="p943314495283"></a><a name="p943314495283"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p18433104911284"><a name="p18433104911284"></a><a name="p18433104911284"></a>A矩阵是否转置的标记。</p>
</td>
</tr>
<tr id="row1247764710288"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p114782471283"><a name="p114782471283"></a><a name="p114782471283"></a>transB</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p3478154710287"><a name="p3478154710287"></a><a name="p3478154710287"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1347884716286"><a name="p1347884716286"></a><a name="p1347884716286"></a>B矩阵是否转置的标记。</p>
</td>
</tr>
<tr id="row173541845172818"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p4354245152812"><a name="p4354245152812"></a><a name="p4354245152812"></a>transC</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p23541345172819"><a name="p23541345172819"></a><a name="p23541345172819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1535416457282"><a name="p1535416457282"></a><a name="p1535416457282"></a>C矩阵的标记，当前仅支持ACL_TRANS_N。</p>
</td>
</tr>
<tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p923414586318"><a name="p923414586318"></a><a name="p923414586318"></a>m</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p22333584316"><a name="p22333584316"></a><a name="p22333584316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p823110581736"><a name="p823110581736"></a><a name="p823110581736"></a>矩阵A的行数与矩阵C的行数。</p>
</td>
</tr>
<tr id="row745163011317"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1545220301039"><a name="p1545220301039"></a><a name="p1545220301039"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p445213305311"><a name="p445213305311"></a><a name="p445213305311"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10452123012318"><a name="p10452123012318"></a><a name="p10452123012318"></a>矩阵B的列数与矩阵C的列数。</p>
</td>
</tr>
<tr id="row152610417522"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p25278455217"><a name="p25278455217"></a><a name="p25278455217"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p3527194125213"><a name="p3527194125213"></a><a name="p3527194125213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1527114115210"><a name="p1527114115210"></a><a name="p1527114115210"></a>矩阵A的列数与矩阵B的行数。</p>
</td>
</tr>
<tr id="row79830322035"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p59833321038"><a name="p59833321038"></a><a name="p59833321038"></a>alpha</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p7983143212316"><a name="p7983143212316"></a><a name="p7983143212316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p12984103217313"><a name="p12984103217313"></a><a name="p12984103217313"></a>用于执行乘操作的标量α的指针。</p>
</td>
</tr>
<tr id="row108515351035"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p168612352316"><a name="p168612352316"></a><a name="p168612352316"></a>matrixA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p88663510319"><a name="p88663510319"></a><a name="p88663510319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p88623513312"><a name="p88623513312"></a><a name="p88623513312"></a>矩阵A的指针。</p>
</td>
</tr>
<tr id="row7600145922816"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1660045932817"><a name="p1660045932817"></a><a name="p1660045932817"></a>lda</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p260018596289"><a name="p260018596289"></a><a name="p260018596289"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1725152063112"><a name="p1725152063112"></a><a name="p1725152063112"></a>A矩阵的主维，此时选择转置，按行优先，则lda为A的列数。预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row1635044115317"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p153511441434"><a name="p153511441434"></a><a name="p153511441434"></a>dataTypeA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1735112415320"><a name="p1735112415320"></a><a name="p1735112415320"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p93511041338"><a name="p93511041338"></a><a name="p93511041338"></a>矩阵A的数据类型。</p>
</td>
</tr>
<tr id="row3823194314315"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p13824194316314"><a name="p13824194316314"></a><a name="p13824194316314"></a>matrixB</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1982454315317"><a name="p1982454315317"></a><a name="p1982454315317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1282418431319"><a name="p1282418431319"></a><a name="p1282418431319"></a>矩阵B的指针。</p>
</td>
</tr>
<tr id="row14646642917"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p134641261290"><a name="p134641261290"></a><a name="p134641261290"></a>ldb</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p154641669290"><a name="p154641669290"></a><a name="p154641669290"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p124642652912"><a name="p124642652912"></a><a name="p124642652912"></a>B矩阵的主维，此时选择转置，按行优先，则ldb为B的列数。预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row1620119387311"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p520119381336"><a name="p520119381336"></a><a name="p520119381336"></a>dataTypeB</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p18201183816315"><a name="p18201183816315"></a><a name="p18201183816315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1320114382313"><a name="p1320114382313"></a><a name="p1320114382313"></a>矩阵B的数据类型。</p>
</td>
</tr>
<tr id="row16553471134"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p166551147135"><a name="p166551147135"></a><a name="p166551147135"></a>beta</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p76554477317"><a name="p76554477317"></a><a name="p76554477317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p465511471933"><a name="p465511471933"></a><a name="p465511471933"></a>用于执行乘操作的标量β的指针。</p>
</td>
</tr>
<tr id="row163443501631"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p4345195015312"><a name="p4345195015312"></a><a name="p4345195015312"></a>matrixC</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1234514507318"><a name="p1234514507318"></a><a name="p1234514507318"></a>输入&amp;输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p23452501136"><a name="p23452501136"></a><a name="p23452501136"></a>矩阵C的指针。</p>
</td>
</tr>
<tr id="row17461613132913"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p20461713182916"><a name="p20461713182916"></a><a name="p20461713182916"></a>ldc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p7461113132914"><a name="p7461113132914"></a><a name="p7461113132914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p446313182916"><a name="p446313182916"></a><a name="p446313182916"></a>C矩阵的主维，预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row145854238"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p6451154839"><a name="p6451154839"></a><a name="p6451154839"></a>dataTypeC</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p6453541630"><a name="p6453541630"></a><a name="p6453541630"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p345154531"><a name="p345154531"></a><a name="p345154531"></a>矩阵C的数据类型。</p>
</td>
</tr>
<tr id="row96189163294"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p16188166296"><a name="p16188166296"></a><a name="p16188166296"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1661811632913"><a name="p1661811632913"></a><a name="p1661811632913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1461811164296"><a name="p1461811164296"></a><a name="p1461811164296"></a>计算精度，默认高精度。</p>
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

## 返回值说明<a name="section14132145124818"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

