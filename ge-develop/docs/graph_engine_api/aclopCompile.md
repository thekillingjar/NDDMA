# aclopCompile<a name="ZH-CN_TOPIC_0000001312481321"></a>

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

编译指定算子。在编译算子前，可以调用[aclSetCompileopt](aclSetCompileopt.md)接口设置编译选项。

每个算子的输入、输出组织不同，在调用接口时严格按照算子输入、输出参数来组织算子，用户在调用aclopCompile时，接口会根据optype、输入tensor的描述、输出tensor的描述、attr等信息查找对应的任务，再编译算子。

## 函数原型<a name="section13230182415108"></a>

```
aclError aclopCompile(const char *opType,
int numInputs,
const aclTensorDesc *const inputDesc[],
int numOutputs,
const aclTensorDesc *const outputDesc[],
const aclopAttr *attr,
aclopEngineType engineType,
aclopCompileType compileFlag,
const char *opPath)
```

## 参数说明<a name="section75395119104"></a>

<a name="table165922618392"></a>
<table><thead align="left"><tr id="row12659192620393"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p96591226173918"><a name="p96591226173918"></a><a name="p96591226173918"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p566014266397"><a name="p566014266397"></a><a name="p566014266397"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p6660152673913"><a name="p6660152673913"></a><a name="p6660152673913"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row46601026153914"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p966032614393"><a name="p966032614393"></a><a name="p966032614393"></a>opType</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p26601126183913"><a name="p26601126183913"></a><a name="p26601126183913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p96601326173911"><a name="p96601326173911"></a><a name="p96601326173911"></a>算子类型名称的指针。</p>
</td>
</tr>
<tr id="row156601426183920"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p11660826173918"><a name="p11660826173918"></a><a name="p11660826173918"></a>numInputs</p>
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
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p61171652213"><a name="p61171652213"></a><a name="p61171652213"></a>算子输入tensor的描述的指针数组。</p>
<p id="p5224192117317"><a name="p5224192117317"></a><a name="p5224192117317"></a>需提前调用<a href="aclCreateTensorDesc.md">aclCreateTensorDesc</a>接口创建aclTensorDesc类型。</p>
<p id="p178282021205414"><a name="p178282021205414"></a><a name="p178282021205414"></a>inputDesc数组中的元素个数必须与numInputs参数值保持一致。</p>
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
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p14900547314"><a name="p14900547314"></a><a name="p14900547314"></a>算子输出tensor的描述的指针数组。</p>
<p id="p240012241933"><a name="p240012241933"></a><a name="p240012241933"></a>需提前调用<a href="aclCreateTensorDesc.md">aclCreateTensorDesc</a>接口创建aclTensorDesc类型。</p>
<p id="p14716113910542"><a name="p14716113910542"></a><a name="p14716113910542"></a>outputDesc数组中的元素个数必须与numOutputs参数值保持一致。</p>
</td>
</tr>
<tr id="row164314237210"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1843118231424"><a name="p1843118231424"></a><a name="p1843118231424"></a>attr</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p84315231327"><a name="p84315231327"></a><a name="p84315231327"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p204315236220"><a name="p204315236220"></a><a name="p204315236220"></a>算子属性的指针。</p>
<p id="p341083814317"><a name="p341083814317"></a><a name="p341083814317"></a>需提前调用<a href="aclopCreateAttr.md">aclopCreateAttr</a>接口创建aclopAttr类型。</p>
</td>
</tr>
<tr id="row344817319212"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p164481531826"><a name="p164481531826"></a><a name="p164481531826"></a>engineType</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p2448183119211"><a name="p2448183119211"></a><a name="p2448183119211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p686145865111"><a name="p686145865111"></a><a name="p686145865111"></a>算子执行引擎。</p>
</td>
</tr>
<tr id="row122676411528"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p182678411722"><a name="p182678411722"></a><a name="p182678411722"></a>compileFlag</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1026717411327"><a name="p1026717411327"></a><a name="p1026717411327"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p8791431194814"><a name="p8791431194814"></a><a name="p8791431194814"></a>算子编译标识。</p>
</td>
</tr>
<tr id="row17904204811915"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p6962105814308"><a name="p6962105814308"></a><a name="p6962105814308"></a>opPath</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p996213585303"><a name="p996213585303"></a><a name="p996213585303"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p199621058143016"><a name="p199621058143016"></a><a name="p199621058143016"></a>算子实现文件（*.py）的路径的指针，不包含文件名。预留参数，当前仅支持设置为nullptr。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section25791320141317"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section57456511778"></a>

-   编译动态Shape的算子时，如果Shape具体值不明确，但Shape范围明确，则在调用aclCreateTensorDesc接口创建[aclTensorDesc](aclTensorDesc.md)类型时，需要将dims数组的表示动态维度的元素值设置为-1，再调用[aclSetTensorShapeRange](aclSetTensorShapeRange.md)接口设置tensor的各个维度的取值范围；如果Shape具体值以及Shape范围都不明确（该场景预留），则在调用aclCreateTensorDesc接口创建[aclTensorDesc](aclTensorDesc.md)类型时，需要将dims数组的值设置为-2，例如：int64\_t dims\[\] = \{-2\}。
-   编译有可选输入的算子时，如果可选输入不使用，则需按此种方式创建[aclTensorDesc](aclTensorDesc.md)类型的数据：[aclCreateTensorDesc](aclCreateTensorDesc.md)\(ACL\_DT\_UNDEFINED, 0, nullptr, ACL\_FORMAT\_UNDEFINED\)，表示数据类型设置为ACL\_DT\_UNDEFINED，Format设置为ACL\_FORMAT\_UNDEFINED，Shape信息为nullptr。
-   编译有constant输入的算子时，需要先调用[aclSetTensorConst](aclSetTensorConst.md)接口设置constant输入。调用aclopCompile接口、[aclopExecuteV2](aclopExecuteV2.md)接口前，设置的constant输入数据必须保持一致。

