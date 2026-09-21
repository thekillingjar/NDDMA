# aclblasCreateHandleForS8gemm<a name="ZH-CN_TOPIC_0000001265081886"></a>

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

创建矩阵-矩阵乘的handle，输入数据的数据类型为int8\_t，输出数据的数据类型为int32\_t。创建handle成功后，需调用[aclopExecWithHandle](aclopExecWithHandle.md)接口执行算子。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclblasCreateHandleForS8gemm(aclTransType transA,
aclTransType transB,
aclTransType transC,
int m,
int n,
int k,
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
<tbody><tr id="row652511911418"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p2043314915286"><a name="p2043314915286"></a><a name="p2043314915286"></a>transA</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p943314495283"><a name="p943314495283"></a><a name="p943314495283"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p18433104911284"><a name="p18433104911284"></a><a name="p18433104911284"></a>A矩阵是否转置的标记。</p>
</td>
</tr>
<tr id="row19824111613147"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p114782471283"><a name="p114782471283"></a><a name="p114782471283"></a>transB</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p3478154710287"><a name="p3478154710287"></a><a name="p3478154710287"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1347884716286"><a name="p1347884716286"></a><a name="p1347884716286"></a>B矩阵是否转置的标记。</p>
</td>
</tr>
<tr id="row5578714121416"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p4354245152812"><a name="p4354245152812"></a><a name="p4354245152812"></a>transC</p>
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
<tr id="row1049132310144"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p16188166296"><a name="p16188166296"></a><a name="p16188166296"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1661811632913"><a name="p1661811632913"></a><a name="p1661811632913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1461811164296"><a name="p1461811164296"></a><a name="p1461811164296"></a>计算精度，默认高精度。</p>
</td>
</tr>
<tr id="row122676411528"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p496419581561"><a name="p496419581561"></a><a name="p496419581561"></a>handle</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p179631658067"><a name="p179631658067"></a><a name="p179631658067"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p994010581068"><a name="p994010581068"></a><a name="p994010581068"></a>“算子执行handle的指针”的指针。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

