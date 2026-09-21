# aclopCompileAndExecute<a name="ZH-CN_TOPIC_0000001312400733"></a>

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

编译并执行指定的算子，当前只支持固定Shape的算子。异步接口。

每个算子的输入、输出、属性不同，接口会根据optype、输入tensor的描述、输出tensor的描述、attr等信息查找对应的任务，并下发执行，因此在调用本接口时需严格按照算子输入、输出、属性来组织算子。

在编译执行算子前，可以调用[aclSetCompileopt](aclSetCompileopt.md)接口设置编译选项。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclopCompileAndExecute(const char *opType,
int numInputs,
const aclTensorDesc *const inputDesc[],
const aclDataBuffer *const inputs[],
int numOutputs,
const aclTensorDesc *const outputDesc[],
aclDataBuffer *const outputs[],
const aclopAttr *attr,
aclopEngineType engineType,
aclopCompileType compileFlag,
const char *opPath,
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
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p0124125211120"><a name="p0124125211120"></a><a name="p0124125211120"></a>opType</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1412319523115"><a name="p1412319523115"></a><a name="p1412319523115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1712295213116"><a name="p1712295213116"></a><a name="p1712295213116"></a>算子类型名称的指针。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1912212521619"><a name="p1912212521619"></a><a name="p1912212521619"></a>numInputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p171219521613"><a name="p171219521613"></a><a name="p171219521613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p212018526117"><a name="p212018526117"></a><a name="p212018526117"></a>算子输入tensor的数量。</p>
</td>
</tr>
<tr id="row137987158341"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p131205521010"><a name="p131205521010"></a><a name="p131205521010"></a>inputDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p4118352412"><a name="p4118352412"></a><a name="p4118352412"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p61171652213"><a name="p61171652213"></a><a name="p61171652213"></a>算子输入tensor描述的指针数组。</p>
<p id="p47921754738"><a name="p47921754738"></a><a name="p47921754738"></a>需提前调用<a href="aclCreateTensorDesc.md">aclCreateTensorDesc</a>接口创建aclTensorDesc类型。</p>
<p id="p33781838193213"><a name="p33781838193213"></a><a name="p33781838193213"></a>inputDesc数组中的元素个数必须与numInputs参数值保持一致，且inputs数组与inputDesc数组中的元素必须一一对应。</p>
</td>
</tr>
<tr id="row17409124312112"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p841019431113"><a name="p841019431113"></a><a name="p841019431113"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p144101343611"><a name="p144101343611"></a><a name="p144101343611"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p12410154317110"><a name="p12410154317110"></a><a name="p12410154317110"></a>算子输入tensor的指针数组。</p>
<p id="p163121023340"><a name="p163121023340"></a><a name="p163121023340"></a>需提前调用<span id="ph0151456133520"><a name="ph0151456133520"></a><a name="ph0151456133520"></a>aclCreateDataBuffer</span>接口创建aclDataBuffer类型的数据。</p>
<p id="p1164444818324"><a name="p1164444818324"></a><a name="p1164444818324"></a>inputs数组中的元素个数必须与numInputs参数值保持一致，且inputs数组与inputDesc数组中的元素必须一一对应。</p>
</td>
</tr>
<tr id="row05461045211"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p154634514113"><a name="p154634514113"></a><a name="p154634514113"></a>numOutputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p3547245119"><a name="p3547245119"></a><a name="p3547245119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1154712459119"><a name="p1154712459119"></a><a name="p1154712459119"></a>算子输出tensor的数量。</p>
</td>
</tr>
<tr id="row17899134718115"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1899947117"><a name="p1899947117"></a><a name="p1899947117"></a>outputDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p88997471018"><a name="p88997471018"></a><a name="p88997471018"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p14900547314"><a name="p14900547314"></a><a name="p14900547314"></a>算子输出tensor描述的指针数组。</p>
<p id="p84651511844"><a name="p84651511844"></a><a name="p84651511844"></a>需提前调用<a href="aclCreateTensorDesc.md">aclCreateTensorDesc</a>接口创建aclTensorDesc类型。</p>
<p id="p4708111119336"><a name="p4708111119336"></a><a name="p4708111119336"></a>outputDesc数组中的元素个数必须与numOutputs参数值保持一致，且outputs数组与outputDesc数组中的元素必须一一对应。</p>
</td>
</tr>
<tr id="row164314237210"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1843118231424"><a name="p1843118231424"></a><a name="p1843118231424"></a>outputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p84315231327"><a name="p84315231327"></a><a name="p84315231327"></a>输入&amp;输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p204315236220"><a name="p204315236220"></a><a name="p204315236220"></a>算子输出tensor的指针数组。</p>
<p id="p102722341412"><a name="p102722341412"></a><a name="p102722341412"></a>需提前调用<span id="ph7174222145312"><a name="ph7174222145312"></a><a name="ph7174222145312"></a>aclCreateDataBuffer</span>接口创建aclDataBuffer类型的数据。</p>
<p id="p1457319556620"><a name="p1457319556620"></a><a name="p1457319556620"></a>outputs数组中的元素个数必须与numOutputs参数值保持一致，且outputs数组与outputDesc数组中的元素必须一一对应。</p>
</td>
</tr>
<tr id="row344817319212"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p164481531826"><a name="p164481531826"></a><a name="p164481531826"></a>attr</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p2448183119211"><a name="p2448183119211"></a><a name="p2448183119211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10448173116210"><a name="p10448173116210"></a><a name="p10448173116210"></a>算子属性的指针。</p>
<p id="p380382161017"><a name="p380382161017"></a><a name="p380382161017"></a>需提前调用<a href="aclopCreateAttr.md">aclopCreateAttr</a>接口创建aclopAttr类型。</p>
</td>
</tr>
<tr id="row581211341425"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p3103161916310"><a name="p3103161916310"></a><a name="p3103161916310"></a>engineType</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p6103121910313"><a name="p6103121910313"></a><a name="p6103121910313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p686145865111"><a name="p686145865111"></a><a name="p686145865111"></a>算子执行引擎。</p>
</td>
</tr>
<tr id="row952920381629"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1110411910313"><a name="p1110411910313"></a><a name="p1110411910313"></a>compileFlag</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p510416191310"><a name="p510416191310"></a><a name="p510416191310"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p8791431194814"><a name="p8791431194814"></a><a name="p8791431194814"></a>算子编译标识。</p>
</td>
</tr>
<tr id="row19579164218218"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p6962105814308"><a name="p6962105814308"></a><a name="p6962105814308"></a>opPath</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p996213585303"><a name="p996213585303"></a><a name="p996213585303"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p145641821772"><a name="p145641821772"></a><a name="p145641821772"></a>算子实现文件（*.py）路径的指针，不包含文件名。预留参数，当前仅支持设置为nullptr。</p>
</td>
</tr>
<tr id="row122676411528"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p182678411722"><a name="p182678411722"></a><a name="p182678411722"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1026717411327"><a name="p1026717411327"></a><a name="p1026717411327"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p2267841420"><a name="p2267841420"></a><a name="p2267841420"></a>该算子需要加载的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section29496561849"></a>

-   多线程场景下，不支持调用本接口时指定同一个Stream或使用默认Stream，否则可能任务执行异常。
-   每个算子的输入、输出、属性不同，需要应用在调用时严格按照算子输入、输出、属性来组织算子，用户在调用aclopCompileAndExecute时，根据optype、输入tensor的描述、输出tensor的描述、attr等信息查找对应的任务，再编译、执行算子。
-   编译、执行有可选输入的算子时，如果可选输入不使用，则：
    -   需按此种方式创建aclTensorDesc类型的数据：[aclCreateTensorDesc](aclCreateTensorDesc.md)\(ACL\_DT\_UNDEFINED, 0, nullptr, ACL\_FORMAT\_UNDEFINED\)，表示数据类型设置为ACL\_DT\_UNDEFINED，Format设置为ACL\_FORMAT\_UNDEFINED，Shape信息为nullptr；
    -   需按此种方式创建aclDataBuffer类型的数据：aclCreateDataBuffer\(nullptr, 0\)，同时aclDataBuffer中的数据不需要释放，因为是空指针。

-   编译、执行有constant输入的算子时，需要先调用[aclSetTensorConst](aclSetTensorConst.md)接口设置constant输入。

    对于算子有constant输入的场景，如果未调用[aclSetTensorConst](aclSetTensorConst.md)接口设置constant输入，则需调用[aclSetTensorPlaceMent](aclSetTensorPlaceMent.md)设置TensorDesc的placement属性，将memType设置为Host内存。

-   在执行单算子时，一般要求用户传入的输入/输出tensor数据是存放在Device内存的，比如两个tensor相加的场景。但是，也存在部分算子，除了将featureMap、weight等Device内存中的tensor数据作为输入外，还需tensor shape信息、learningRate、tensor的dims信息等通常在Host内存中的tensor数据也作为输入，此时，用户不需要额外将这些Host内存中的tensor数据拷贝到Device上作为输入，只需要调用[aclSetTensorPlaceMent](aclSetTensorPlaceMent.md)接口设置对应TensorDesc的placement属性为Host内存，在执行单算子时，设置了placement属性为Host内存的TensorDesc，其对应的tensor数据必须存放在Host内存中，接口内部会自动将Host上的tensor数据拷贝到Device上。

