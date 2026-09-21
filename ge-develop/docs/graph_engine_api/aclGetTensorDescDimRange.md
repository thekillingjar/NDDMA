# aclGetTensorDescDimRange<a name="ZH-CN_TOPIC_0000001265081418"></a>

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

## 功能说明<a name="section1661552975617"></a>

获取tensor描述中指定维度的范围，\[1,-1\]表示全shape范围。

当[aclGetTensorDescNumDims](aclGetTensorDescNumDims.md)接口的返回值为ACL\_UNKNOWN\_RANK时，表示动态Shape场景下维度个数未知，则不能调用aclGetTensorDescDimRange接口获取指定维度的范围。

## 函数原型<a name="section15693534145616"></a>

```
aclError aclGetTensorDescDimRange(const aclTensorDesc *desc, size_t index, size_t dimRangeNum, int64_t *dimRange)
```

## 参数说明<a name="section14811440155616"></a>

<a name="table15185939162113"></a>
<table><thead align="left"><tr id="row818513911216"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p20185103918218"><a name="p20185103918218"></a><a name="p20185103918218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p11185143916215"><a name="p11185143916215"></a><a name="p11185143916215"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p191851939172115"><a name="p191851939172115"></a><a name="p191851939172115"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row9185133902114"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p171851539152115"><a name="p171851539152115"></a><a name="p171851539152115"></a>desc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p3185639132117"><a name="p3185639132117"></a><a name="p3185639132117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p2648133315150"><a name="p2648133315150"></a><a name="p2648133315150"></a>aclTensorDesc类型的指针。</p>
<p id="p1861313125319"><a name="p1861313125319"></a><a name="p1861313125319"></a>需提前调用<a href="aclCreateTensorDesc.md">aclCreateTensorDesc</a>接口创建aclTensorDesc类型。</p>
</td>
</tr>
<tr id="row10284524113410"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p82851424163417"><a name="p82851424163417"></a><a name="p82851424163417"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p11285142403411"><a name="p11285142403411"></a><a name="p11285142403411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1728522417343"><a name="p1728522417343"></a><a name="p1728522417343"></a>指定获取第几个维度的大小，index值从0开始。</p>
<p id="p10375141241215"><a name="p10375141241215"></a><a name="p10375141241215"></a>用户调用<a href="aclGetTensorDescNumDims.md">aclGetTensorDescNumDims</a>接口获取shape维度个数，这个Index的取值范围：[0, (shape维度个数-1)]。</p>
</td>
</tr>
<tr id="row185651344145417"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1565174418543"><a name="p1565174418543"></a><a name="p1565174418543"></a>dimRangeNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p35651944135419"><a name="p35651944135419"></a><a name="p35651944135419"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p145661844105414"><a name="p145661844105414"></a><a name="p145661844105414"></a>dimRange的长度，该值必须大于等于2。</p>
</td>
</tr>
<tr id="row39911581442"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p131009585419"><a name="p131009585419"></a><a name="p131009585419"></a>dimRange</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1010035815413"><a name="p1010035815413"></a><a name="p1010035815413"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p21001558943"><a name="p21001558943"></a><a name="p21001558943"></a>tensor描述中index指定维度的Shape范围。</p>
<p id="p621211429420"><a name="p621211429420"></a><a name="p621211429420"></a>dimRange是一个数组，数组的第一个元素值表示Shape范围的最小值，第二个元素值表示Shape范围的最大值。该数组中仅前2个元素值有效。</p>
<p id="p1784418321749"><a name="p1784418321749"></a><a name="p1784418321749"></a>当dimRange数组的值为[1,-1]时，表示全Shape范围。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25151444115613"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

