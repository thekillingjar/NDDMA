# aclCreateTensorDesc<a name="ZH-CN_TOPIC_0000001265400522"></a>

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

创建aclTensorDesc类型的数据，该数据类型用于描述tensor的数据类型、shape、format等信息。

如需销毁aclTensorDesc类型的数据，请参见[aclDestroyTensorDesc](aclDestroyTensorDesc.md)。

## 函数原型<a name="section15693534145616"></a>

```
aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
int numDims,
const int64_t *dims,
aclFormat format)
```

## 参数说明<a name="section14811440155616"></a>

<a name="table16824183151019"></a>
<table><thead align="left"><tr id="row1282411341019"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p58241934109"><a name="p58241934109"></a><a name="p58241934109"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p118247315102"><a name="p118247315102"></a><a name="p118247315102"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p1382413151019"><a name="p1382413151019"></a><a name="p1382413151019"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row2082493121017"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1270714236122"><a name="p1270714236122"></a><a name="p1270714236122"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p68251535100"><a name="p68251535100"></a><a name="p68251535100"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p34051452111212"><a name="p34051452111212"></a><a name="p34051452111212"></a>tensor描述的数据类型。</p>
</td>
</tr>
<tr id="row1351698181015"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p770514234128"><a name="p770514234128"></a><a name="p770514234128"></a>numDims</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p65163821019"><a name="p65163821019"></a><a name="p65163821019"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p11971152171214"><a name="p11971152171214"></a><a name="p11971152171214"></a>tensor描述shape的维度个数。</p>
<p id="p275584014394"><a name="p275584014394"></a><a name="p275584014394"></a>numDims存在以下约束：</p>
<a name="ul14755184033910"></a><a name="ul14755184033910"></a><ul id="ul14755184033910"><li>numDims必须大于或等于0；</li><li>numDims&gt;0时，numDims的值必须与dims数组的长度保持一致；</li><li>numDims=0时，系统不使用dims数组值，dims参数值无效。</li></ul>
</td>
</tr>
<tr id="row9574161561213"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1157517157125"><a name="p1157517157125"></a><a name="p1157517157125"></a>dims</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1657515158127"><a name="p1657515158127"></a><a name="p1657515158127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p145756153129"><a name="p145756153129"></a><a name="p145756153129"></a>tensor描述维度大小的指针。</p>
<p id="p176141119123511"><a name="p176141119123511"></a><a name="p176141119123511"></a>dims是一个数组，数组中的每个元素表示tensor中每个维度的大小，如果数组中某个元素的值为0，则为空tensor。</p>
<p id="p6101121110310"><a name="p6101121110310"></a><a name="p6101121110310"></a>如果用户需要使用空tensor，则在申请内存时，内存大小最小为1Byte，以保障后续业务正常运行。</p>
</td>
</tr>
<tr id="row1099711714123"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p7997131711211"><a name="p7997131711211"></a><a name="p7997131711211"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1599711178129"><a name="p1599711178129"></a><a name="p1599711178129"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p599701712124"><a name="p599701712124"></a><a name="p599701712124"></a>tensor描述的format。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25151444115613"></a>

返回aclTensorDesc类型的指针。

