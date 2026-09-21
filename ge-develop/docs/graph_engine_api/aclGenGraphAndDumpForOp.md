# aclGenGraphAndDumpForOp<a name="ZH-CN_TOPIC_0000001312721653"></a>

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

构建单个算子的图并Dump，生成.txt文件。此处的图是指表达算子IR（Intermediate Representation）的结构，图编译过程中由于融合、拆分等原因会导致图不断变换、进而生成新的图，这些变换的目的是为了将用户表达的图适配到昇腾AI处理器上，获得更高性能。

**使用场景**：算子调优场景下，调用本接口构建单个算子的图并Dump，生成.txt文件，作为算子调优的输入数据之一。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclGenGraphAndDumpForOp(const char *opType,
int numInputs,
const aclTensorDesc *const inputDesc[],
const aclDataBuffer *const inputs[],
int numOutputs,
const aclTensorDesc *const outputDesc[],
aclDataBuffer *const outputs[],
const aclopAttr *attr,
aclopEngineType engineType,
const char *graphDumpPath,
const aclGraphDumpOption *graphDumpOpt)
```

## 参数说明<a name="section75395119104"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p0124125211120"><a name="p0124125211120"></a><a name="p0124125211120"></a>opType</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1412319523115"><a name="p1412319523115"></a><a name="p1412319523115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p1712295213116"><a name="p1712295213116"></a><a name="p1712295213116"></a>指定算子类型名称的指针。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1912212521619"><a name="p1912212521619"></a><a name="p1912212521619"></a>numInputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p171219521613"><a name="p171219521613"></a><a name="p171219521613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p212018526117"><a name="p212018526117"></a><a name="p212018526117"></a>算子输入tensor的数量。</p>
</td>
</tr>
<tr id="row137987158341"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p131205521010"><a name="p131205521010"></a><a name="p131205521010"></a>inputDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p4118352412"><a name="p4118352412"></a><a name="p4118352412"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p61171652213"><a name="p61171652213"></a><a name="p61171652213"></a>算子输入tensor描述的指针数组。</p>
<p id="p47921754738"><a name="p47921754738"></a><a name="p47921754738"></a>需提前调用<a href="aclCreateTensorDesc.md">aclCreateTensorDesc</a>接口创建aclTensorDesc类型。</p>
<p id="p33781838193213"><a name="p33781838193213"></a><a name="p33781838193213"></a>inputDesc数组中的元素个数必须与numInputs参数值保持一致，且inputs数组与inputDesc数组中的元素必须一一对应。</p>
</td>
</tr>
<tr id="row17409124312112"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p841019431113"><a name="p841019431113"></a><a name="p841019431113"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p144101343611"><a name="p144101343611"></a><a name="p144101343611"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p12410154317110"><a name="p12410154317110"></a><a name="p12410154317110"></a>算子输入tensor的指针数组。</p>
<p id="p163121023340"><a name="p163121023340"></a><a name="p163121023340"></a>需提前调用<span id="ph155001119572"><a name="ph155001119572"></a><a name="ph155001119572"></a>aclCreateDataBuffer</span>接口创建aclDataBuffer类型的数据。</p>
<p id="p1164444818324"><a name="p1164444818324"></a><a name="p1164444818324"></a>inputs数组中的元素个数必须与numInputs参数值保持一致，且inputs数组与inputDesc数组中的元素必须一一对应。</p>
</td>
</tr>
<tr id="row05461045211"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p154634514113"><a name="p154634514113"></a><a name="p154634514113"></a>numOutputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p3547245119"><a name="p3547245119"></a><a name="p3547245119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p1154712459119"><a name="p1154712459119"></a><a name="p1154712459119"></a>算子输出tensor的数量。</p>
</td>
</tr>
<tr id="row17899134718115"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1899947117"><a name="p1899947117"></a><a name="p1899947117"></a>outputDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p88997471018"><a name="p88997471018"></a><a name="p88997471018"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p14900547314"><a name="p14900547314"></a><a name="p14900547314"></a>算子输出tensor描述的指针数组。</p>
<p id="p84651511844"><a name="p84651511844"></a><a name="p84651511844"></a>需提前调用<a href="aclCreateTensorDesc.md">aclCreateTensorDesc</a>接口创建aclTensorDesc类型。</p>
<p id="p4708111119336"><a name="p4708111119336"></a><a name="p4708111119336"></a>outputDesc数组中的元素个数必须与numOutputs参数值保持一致，且outputs数组与outputDesc数组中的元素必须一一对应。</p>
</td>
</tr>
<tr id="row164314237210"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1843118231424"><a name="p1843118231424"></a><a name="p1843118231424"></a>outputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p84315231327"><a name="p84315231327"></a><a name="p84315231327"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p204315236220"><a name="p204315236220"></a><a name="p204315236220"></a>算子输出tensor的指针数组。</p>
<p id="p102722341412"><a name="p102722341412"></a><a name="p102722341412"></a>需提前调用<span id="ph0151456133520"><a name="ph0151456133520"></a><a name="ph0151456133520"></a>aclCreateDataBuffer</span>接口创建aclDataBuffer类型的数据。</p>
<p id="p1457319556620"><a name="p1457319556620"></a><a name="p1457319556620"></a>outputs数组中的元素个数必须与numOutputs参数值保持一致，且outputs数组与outputDesc数组中的元素必须一一对应。</p>
</td>
</tr>
<tr id="row344817319212"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p164481531826"><a name="p164481531826"></a><a name="p164481531826"></a>attr</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p2448183119211"><a name="p2448183119211"></a><a name="p2448183119211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p10448173116210"><a name="p10448173116210"></a><a name="p10448173116210"></a>算子属性的指针。</p>
<p id="p380382161017"><a name="p380382161017"></a><a name="p380382161017"></a>需提前调用<a href="aclopCreateAttr.md">aclopCreateAttr</a>接口创建aclopAttr类型。</p>
</td>
</tr>
<tr id="row581211341425"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p3103161916310"><a name="p3103161916310"></a><a name="p3103161916310"></a>engineType</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p6103121910313"><a name="p6103121910313"></a><a name="p6103121910313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p686145865111"><a name="p686145865111"></a><a name="p686145865111"></a>算子执行引擎。</p>
</td>
</tr>
<tr id="row952920381629"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1110411910313"><a name="p1110411910313"></a><a name="p1110411910313"></a>graphDumpPath</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p510416191310"><a name="p510416191310"></a><a name="p510416191310"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p1476966174917"><a name="p1476966174917"></a><a name="p1476966174917"></a>Dump单算子图所生成的文件（文件名后缀为.txt）的保存路径的指针。</p>
<p id="p179673232814"><a name="p179673232814"></a><a name="p179673232814"></a>路径需要由用户提前创建，路径中如果不指定文件名，则由接口内部指定文件名。</p>
<p id="p1443412116313"><a name="p1443412116313"></a><a name="p1443412116313"></a>如果多次Dump时，指定同一个路径，且文件名相同，则最新一次的文件会覆盖之前的文件。</p>
</td>
</tr>
<tr id="row19579164218218"><td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p6962105814308"><a name="p6962105814308"></a><a name="p6962105814308"></a>graphDumpOpt</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p996213585303"><a name="p996213585303"></a><a name="p996213585303"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72%" headers="mcps1.1.4.1.3 "><p id="p13780112614917"><a name="p13780112614917"></a><a name="p13780112614917"></a>单算子图的dump选项的指针。</p>
<p id="p09762041124416"><a name="p09762041124416"></a><a name="p09762041124416"></a>当前该参数预留，仅支持传入nullptr。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

