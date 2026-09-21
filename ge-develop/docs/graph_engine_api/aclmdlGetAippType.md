# aclmdlGetAippType<a name="ZH-CN_TOPIC_0000001312400881"></a>

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

## 功能说明<a name="section36583473819"></a>

获取指定模型的指定输入所支持的AIPP类型（动态AIPP或静态AIPP）及动态AIPP输入对应的index值。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclmdlGetAippType(uint32_t modelId, size_t index, aclmdlInputAippType *type, size_t *dynamicAttachedDataIndex)
```

## 参数说明<a name="section75395119104"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="26.919999999999998%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.46%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="60.62%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="26.919999999999998%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>modelId</p>
</td>
<td class="cellrowborder" valign="top" width="12.46%" headers="mcps1.1.4.1.2 "><p id="p16390135183518"><a name="p16390135183518"></a><a name="p16390135183518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="60.62%" headers="mcps1.1.4.1.3 "><p id="p4388859358"><a name="p4388859358"></a><a name="p4388859358"></a>指定模型的ID。</p>
<p id="p57291851112517"><a name="p57291851112517"></a><a name="p57291851112517"></a>调用<a href="aclmdlLoadFromFile.md">aclmdlLoadFromFile</a>接口/<a href="aclmdlLoadFromMem.md">aclmdlLoadFromMem</a>接口/<a href="aclmdlLoadFromFileWithMem.md">aclmdlLoadFromFileWithMem</a>接口/<a href="aclmdlLoadFromMemWithMem.md">aclmdlLoadFromMemWithMem</a>接口加载模型成功后，会返回模型ID。</p>
</td>
</tr>
<tr id="row1865008162612"><td class="cellrowborder" valign="top" width="26.919999999999998%" headers="mcps1.1.4.1.1 "><p id="p12651118102619"><a name="p12651118102619"></a><a name="p12651118102619"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="12.46%" headers="mcps1.1.4.1.2 "><p id="p86511687266"><a name="p86511687266"></a><a name="p86511687266"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="60.62%" headers="mcps1.1.4.1.3 "><p id="p4651086263"><a name="p4651086263"></a><a name="p4651086263"></a>模型中输入的index。</p>
</td>
</tr>
<tr id="row1268411542811"><td class="cellrowborder" valign="top" width="26.919999999999998%" headers="mcps1.1.4.1.1 "><p id="p1168420516285"><a name="p1168420516285"></a><a name="p1168420516285"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="12.46%" headers="mcps1.1.4.1.2 "><p id="p668518511281"><a name="p668518511281"></a><a name="p668518511281"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="60.62%" headers="mcps1.1.4.1.3 "><p id="p2529402013"><a name="p2529402013"></a><a name="p2529402013"></a>指定模型输入的AIPP类型的指针。</p>
</td>
</tr>
<tr id="row169022003324"><td class="cellrowborder" valign="top" width="26.919999999999998%" headers="mcps1.1.4.1.1 "><p id="p1390220093219"><a name="p1390220093219"></a><a name="p1390220093219"></a>dynamicAttachedDataIndex</p>
</td>
<td class="cellrowborder" valign="top" width="12.46%" headers="mcps1.1.4.1.2 "><p id="p69022033213"><a name="p69022033213"></a><a name="p69022033213"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="60.62%" headers="mcps1.1.4.1.3 "><p id="p8903101327"><a name="p8903101327"></a><a name="p8903101327"></a>当type不为ACL_DATA_WITH_DYNAMIC_AIPP时，该值返回0xFFFFFFFF，表示无效。</p>
<p id="p025421674920"><a name="p025421674920"></a><a name="p025421674920"></a>当type为ACL_DATA_WITH_DYNAMIC_AIPP时，该值返回动态AIPP输入（用于配置动态AIPP参数）的index。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

