# aclmdlCreateAndGetOpDesc<a name="ZH-CN_TOPIC_0000001265400546"></a>

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

## 功能说明<a name="section166231326115013"></a>

获取指定算子的描述信息，包括算子名、输入tensor描述、输出tensor描述，如果查询不到指定算子，则返回报错。该接口不支持动态Shape场景。

**使用场景举例**：执行整网模型推理时，如果产生AI Core报错，可以调用本接口获取报错算子的描述信息，再做进一步错误排查。

**推荐的接口调用顺序如下：**

1.  定义并实现异常回调函数fn\(aclrtExceptionInfoCallback类型\)，回调函数原型请参见aclrtSetExceptionInfoCallback  。

    实现回调函数的关键步骤如下：

    1.  在异常回调函数fn内调用aclrtGetDeviceIdFromExceptionInfo、aclrtGetStreamIdFromExceptionInfo、aclrtGetTaskIdFromExceptionInfo接口分别获取Device ID、Stream ID、Task ID。
    2.  在异常回调函数fn内调用[aclmdlCreateAndGetOpDesc](aclmdlCreateAndGetOpDesc.md)接口获取算子的描述信息。
    3.  在异常回调函数fn内调用[aclGetTensorDescByIndex](aclGetTensorDescByIndex.md)接口获取指定算子输入/输出的tensor描述。
    4.  在异常回调函数fn内调用如下接口获取tensor描述中的数据，进行进一步分析。

        例如，调用[aclGetTensorDescAddress](aclGetTensorDescAddress.md)接口获取tensor数据的内存地址（用户可从该内存地址中获取tensor数据）、调用[aclGetTensorDescType](aclGetTensorDescType.md)接口获取tensor描述中的数据类型、调用[aclGetTensorDescFormat](aclGetTensorDescFormat.md)接口获取tensor描述中的Format、调用[aclGetTensorDescNumDims](aclGetTensorDescNumDims.md)接口获取tensor描述中的Shape维度个数、调用[aclGetTensorDescDimV2](aclGetTensorDescDimV2.md)接口获取Shape中指定维度的大小。

2.  调用aclrtSetExceptionInfoCallback接口设置异常回调函数。
3.  执行模型推理。

    如果存在AI Core报错，则触发回调函数fn，获取算子的信息，进行进一步分析。

## 函数原型<a name="section244219302509"></a>

```
aclError aclmdlCreateAndGetOpDesc(uint32_t deviceId, uint32_t streamId, uint32_t taskId, char *opName, size_t opNameLen, aclTensorDesc **inputDesc, size_t *numInputs, aclTensorDesc **outputDesc, size_t *numOutputs)
```

## 参数说明<a name="section163301034105012"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>deviceId</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p16390135183518"><a name="p16390135183518"></a><a name="p16390135183518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p85521840204019"><a name="p85521840204019"></a><a name="p85521840204019"></a>Device ID。</p>
<p id="p12722648144015"><a name="p12722648144015"></a><a name="p12722648144015"></a>调用<span id="ph221816392579"><a name="ph221816392579"></a><a name="ph221816392579"></a>aclrtGetDeviceIdFromExceptionInfo</span>接口获取异常信息中的Device ID，作为本接口的输入。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p19341911481"><a name="p19341911481"></a><a name="p19341911481"></a>streamId</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p69341916814"><a name="p69341916814"></a><a name="p69341916814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1753116407407"><a name="p1753116407407"></a><a name="p1753116407407"></a>Stream ID。</p>
<p id="p436715215420"><a name="p436715215420"></a><a name="p436715215420"></a>调用<span id="ph1097704614571"><a name="ph1097704614571"></a><a name="ph1097704614571"></a>aclrtGetStreamIdFromExceptionInfo</span>接口获取异常信息中的Stream ID，作为本接口的输入。</p>
</td>
</tr>
<tr id="row147732496171"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p072371917818"><a name="p072371917818"></a><a name="p072371917818"></a>taskId</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p177231319387"><a name="p177231319387"></a><a name="p177231319387"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10511134019402"><a name="p10511134019402"></a><a name="p10511134019402"></a>Task ID。</p>
<p id="p025371864318"><a name="p025371864318"></a><a name="p025371864318"></a>调用<span id="ph141714552578"><a name="ph141714552578"></a><a name="ph141714552578"></a>aclrtGetTaskIdFromExceptionInfo</span>接口获取异常信息中的Task ID，作为本接口的输入。</p>
</td>
</tr>
<tr id="row2604183119586"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p12605123125813"><a name="p12605123125813"></a><a name="p12605123125813"></a>opName</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p18605153110585"><a name="p18605153110585"></a><a name="p18605153110585"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p164957402401"><a name="p164957402401"></a><a name="p164957402401"></a>算子名称字符串的指针。</p>
</td>
</tr>
<tr id="row19160154611389"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1616094610380"><a name="p1616094610380"></a><a name="p1616094610380"></a>opNameLen</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p17160546113812"><a name="p17160546113812"></a><a name="p17160546113812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p11160204613386"><a name="p11160204613386"></a><a name="p11160204613386"></a>算子名称字符串长度。</p>
<p id="p15981547124716"><a name="p15981547124716"></a><a name="p15981547124716"></a>若用户指定的长度比实际算子名称的长度短，则返回报错。</p>
</td>
</tr>
<tr id="row18593514382"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p3597513387"><a name="p3597513387"></a><a name="p3597513387"></a>inputDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p186025143816"><a name="p186025143816"></a><a name="p186025143816"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p960951153818"><a name="p960951153818"></a><a name="p960951153818"></a>算子所有输入的tensor描述的指针，指向一块连续内存的首地址。</p>
</td>
</tr>
<tr id="row13104165614383"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p191043560388"><a name="p191043560388"></a><a name="p191043560388"></a>numInputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1410413565385"><a name="p1410413565385"></a><a name="p1410413565385"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p111050564385"><a name="p111050564385"></a><a name="p111050564385"></a>输入个数的指针。</p>
</td>
</tr>
<tr id="row45871228113913"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p17587172823912"><a name="p17587172823912"></a><a name="p17587172823912"></a>outputDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1458718284390"><a name="p1458718284390"></a><a name="p1458718284390"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p165871428103917"><a name="p165871428103917"></a><a name="p165871428103917"></a>算子所有输出的tensor描述的指针，指向一块连续内存的首地址。</p>
</td>
</tr>
<tr id="row475163715394"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p5751173753913"><a name="p5751173753913"></a><a name="p5751173753913"></a>numOutputs</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p9751437183914"><a name="p9751437183914"></a><a name="p9751437183914"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1675193719399"><a name="p1675193719399"></a><a name="p1675193719399"></a>输出个数的指针。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section7216174405014"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

