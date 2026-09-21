# aclblasS8gemm<a name="ZH-CN_TOPIC_0000001265400770"></a>

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

执行矩阵-矩阵的乘法，C = αAB + βC，输入数据的数据类型为int8\_t，输出数据的数据类型为int32\_t。异步接口。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclblasS8gemm(aclTransType transA,
aclTransType transB,
aclTransType transC,
int m,
int n,
int k,
const int32_t *alpha,
const int8_t *matrixA,
int lda,
const int8_t *matrixB,
int ldb,
const int32_t *beta,
int32_t *matrixC,
int ldc,
aclComputeType type,
aclrtStream stream)
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
<tbody><tr id="row243332813113"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p2043314915286"><a name="p2043314915286"></a><a name="p2043314915286"></a>transA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p943314495283"><a name="p943314495283"></a><a name="p943314495283"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p18433104911284"><a name="p18433104911284"></a><a name="p18433104911284"></a>A矩阵是否转置的标记。</p>
</td>
</tr>
<tr id="row13167926191118"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p114782471283"><a name="p114782471283"></a><a name="p114782471283"></a>transB</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p3478154710287"><a name="p3478154710287"></a><a name="p3478154710287"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1347884716286"><a name="p1347884716286"></a><a name="p1347884716286"></a>B矩阵是否转置的标记。</p>
</td>
</tr>
<tr id="row714242471113"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p4354245152812"><a name="p4354245152812"></a><a name="p4354245152812"></a>transC</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p23541345172819"><a name="p23541345172819"></a><a name="p23541345172819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1535416457282"><a name="p1535416457282"></a><a name="p1535416457282"></a>C矩阵的标记，当前仅支持ACL_TRANS_N。</p>
</td>
</tr>
<tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p2982135817616"><a name="p2982135817616"></a><a name="p2982135817616"></a>m</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p59813581663"><a name="p59813581663"></a><a name="p59813581663"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p129801158560"><a name="p129801158560"></a><a name="p129801158560"></a>矩阵A的行数与矩阵C的行数。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p199798581861"><a name="p199798581861"></a><a name="p199798581861"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1897918581669"><a name="p1897918581669"></a><a name="p1897918581669"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10978205810619"><a name="p10978205810619"></a><a name="p10978205810619"></a>矩阵B的列数与矩阵C的列数。</p>
</td>
</tr>
<tr id="row137987158341"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p11978155817615"><a name="p11978155817615"></a><a name="p11978155817615"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1297716581067"><a name="p1297716581067"></a><a name="p1297716581067"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p49774585613"><a name="p49774585613"></a><a name="p49774585613"></a>矩阵A的列数与矩阵B的行数。</p>
</td>
</tr>
<tr id="row17409124312112"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p49769581260"><a name="p49769581260"></a><a name="p49769581260"></a>alpha</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1797635817616"><a name="p1797635817616"></a><a name="p1797635817616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p189756582068"><a name="p189756582068"></a><a name="p189756582068"></a>用于执行乘操作的标量α的指针。</p>
</td>
</tr>
<tr id="row05461045211"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p89741258165"><a name="p89741258165"></a><a name="p89741258165"></a>matrixA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p119735581267"><a name="p119735581267"></a><a name="p119735581267"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p139731758764"><a name="p139731758764"></a><a name="p139731758764"></a>矩阵A的指针。</p>
</td>
</tr>
<tr id="row1587675411119"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1660045932817"><a name="p1660045932817"></a><a name="p1660045932817"></a>lda</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p260018596289"><a name="p260018596289"></a><a name="p260018596289"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1725152063112"><a name="p1725152063112"></a><a name="p1725152063112"></a>A矩阵的主维，此时选择转置，按行优先，则lda为A的列数。预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row17899134718115"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p49723581162"><a name="p49723581162"></a><a name="p49723581162"></a>matrixB</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1397175812619"><a name="p1397175812619"></a><a name="p1397175812619"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p189705585618"><a name="p189705585618"></a><a name="p189705585618"></a>矩阵B的指针。</p>
</td>
</tr>
<tr id="row7615135851117"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p134641261290"><a name="p134641261290"></a><a name="p134641261290"></a>ldb</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p154641669290"><a name="p154641669290"></a><a name="p154641669290"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p124642652912"><a name="p124642652912"></a><a name="p124642652912"></a>B矩阵的主维，此时选择转置，按行优先，则ldb为B的列数。预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row164314237210"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p13969758766"><a name="p13969758766"></a><a name="p13969758766"></a>beta</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p09695589619"><a name="p09695589619"></a><a name="p09695589619"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p19686584612"><a name="p19686584612"></a><a name="p19686584612"></a>用于执行乘操作的标量β的指针。</p>
</td>
</tr>
<tr id="row344817319212"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p59671158461"><a name="p59671158461"></a><a name="p59671158461"></a>matrixC</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p196675814611"><a name="p196675814611"></a><a name="p196675814611"></a>输入&amp;输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p199659582619"><a name="p199659582619"></a><a name="p199659582619"></a>矩阵C的指针。</p>
</td>
</tr>
<tr id="row1928617561216"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p20461713182916"><a name="p20461713182916"></a><a name="p20461713182916"></a>ldc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p7461113132914"><a name="p7461113132914"></a><a name="p7461113132914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p446313182916"><a name="p446313182916"></a><a name="p446313182916"></a>C矩阵的主维，预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row1967119131217"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p16188166296"><a name="p16188166296"></a><a name="p16188166296"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1661811632913"><a name="p1661811632913"></a><a name="p1661811632913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1461811164296"><a name="p1461811164296"></a><a name="p1461811164296"></a>计算精度，默认高精度。</p>
</td>
</tr>
<tr id="row122676411528"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p496419581561"><a name="p496419581561"></a><a name="p496419581561"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p179631658067"><a name="p179631658067"></a><a name="p179631658067"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p994010581068"><a name="p994010581068"></a><a name="p994010581068"></a>执行算子所在的Stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

