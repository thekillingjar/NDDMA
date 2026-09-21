# aclblasHgemv<a name="ZH-CN_TOPIC_0000001312721905"></a>

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

执行矩阵-向量的乘法，y = αAx + βy，输入数据和输出数据的数据类型为aclFloat16**。**异步接口。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclblasHgemv(aclTransType transA,
int m,
int n,
const aclFloat16 *alpha,
const aclFloat16 *a,
int lda,
const aclFloat16 *x,
int incx,
const aclFloat16 *beta,
aclFloat16 *y,
int incy,
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
<tbody><tr id="row148117424513"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1269164421511"><a name="p1269164421511"></a><a name="p1269164421511"></a>transA</p>
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
<tr id="row137987158341"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p18581111021020"><a name="p18581111021020"></a><a name="p18581111021020"></a>alpha</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1297716581067"><a name="p1297716581067"></a><a name="p1297716581067"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p49774585613"><a name="p49774585613"></a><a name="p49774585613"></a>用于执行乘操作的标量α的指针。</p>
</td>
</tr>
<tr id="row17409124312112"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p6582121019102"><a name="p6582121019102"></a><a name="p6582121019102"></a>a</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1797635817616"><a name="p1797635817616"></a><a name="p1797635817616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p189756582068"><a name="p189756582068"></a><a name="p189756582068"></a>矩阵A的指针。</p>
</td>
</tr>
<tr id="row55221441125111"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p9523144125120"><a name="p9523144125120"></a><a name="p9523144125120"></a>lda</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p119027016167"><a name="p119027016167"></a><a name="p119027016167"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p690220161611"><a name="p690220161611"></a><a name="p690220161611"></a>A矩阵的主维，此时选择转置，按行优先，则lda为A的列数。预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row05461045211"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p658218105105"><a name="p658218105105"></a><a name="p658218105105"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p119735581267"><a name="p119735581267"></a><a name="p119735581267"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p139731758764"><a name="p139731758764"></a><a name="p139731758764"></a>向量x的指针。</p>
</td>
</tr>
<tr id="row4517152595219"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p3825123315162"><a name="p3825123315162"></a><a name="p3825123315162"></a>incx</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p48251533191613"><a name="p48251533191613"></a><a name="p48251533191613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p38257337169"><a name="p38257337169"></a><a name="p38257337169"></a>x连续元素之间的步长。</p>
<p id="p385784454714"><a name="p385784454714"></a><a name="p385784454714"></a>预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row17899134718115"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p9582111017109"><a name="p9582111017109"></a><a name="p9582111017109"></a>beta</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1397175812619"><a name="p1397175812619"></a><a name="p1397175812619"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p189705585618"><a name="p189705585618"></a><a name="p189705585618"></a>用于执行乘操作的标量β的指针。</p>
</td>
</tr>
<tr id="row164314237210"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p9582171011015"><a name="p9582171011015"></a><a name="p9582171011015"></a>y</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p09695589619"><a name="p09695589619"></a><a name="p09695589619"></a>输入&amp;输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p19686584612"><a name="p19686584612"></a><a name="p19686584612"></a>向量y的指针。</p>
</td>
</tr>
<tr id="row11693193615211"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1647547191620"><a name="p1647547191620"></a><a name="p1647547191620"></a>incy</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p20647154781616"><a name="p20647154781616"></a><a name="p20647154781616"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1364724712167"><a name="p1364724712167"></a><a name="p1364724712167"></a>y连续元素之间的步长。</p>
<p id="p19456947184719"><a name="p19456947184719"></a><a name="p19456947184719"></a>预留参数，当前只能设置为-1。</p>
</td>
</tr>
<tr id="row205311442524"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p749715061712"><a name="p749715061712"></a><a name="p749715061712"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p134970011178"><a name="p134970011178"></a><a name="p134970011178"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p174976021713"><a name="p174976021713"></a><a name="p174976021713"></a>计算精度，默认高精度。</p>
</td>
</tr>
<tr id="row344817319212"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p155821010201017"><a name="p155821010201017"></a><a name="p155821010201017"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p196675814611"><a name="p196675814611"></a><a name="p196675814611"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p199659582619"><a name="p199659582619"></a><a name="p199659582619"></a>执行算子所在的Stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

