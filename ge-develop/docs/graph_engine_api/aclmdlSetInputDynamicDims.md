# aclmdlSetInputDynamicDims<a name="ZH-CN_TOPIC_0000001265081778"></a>

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

## 功能说明<a name="section166231326115013"></a>

如果模型输入的Shape是动态的、输入数据Format为ND格式（ND表示支持任意格式），在模型执行前调用本接口设置模型推理时具体维度的值。

## 函数原型<a name="section244219302509"></a>

```
aclError aclmdlSetInputDynamicDims(uint32_t modelId, aclmdlDataset *dataset, size_t index, const aclmdlIODims *dims)
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
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>modelId</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p16390135183518"><a name="p16390135183518"></a><a name="p16390135183518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p8997624142310"><a name="p8997624142310"></a><a name="p8997624142310"></a>模型ID。</p>
<p id="p57291851112517"><a name="p57291851112517"></a><a name="p57291851112517"></a>调用<a href="aclmdlLoadFromFile.md">aclmdlLoadFromFile</a>接口/<a href="aclmdlLoadFromMem.md">aclmdlLoadFromMem</a>接口/<a href="aclmdlLoadFromFileWithMem.md">aclmdlLoadFromFileWithMem</a>接口/<a href="aclmdlLoadFromMemWithMem.md">aclmdlLoadFromMemWithMem</a>接口加载模型成功后，会返回模型ID。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p19341911481"><a name="p19341911481"></a><a name="p19341911481"></a>dataset</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p69341916814"><a name="p69341916814"></a><a name="p69341916814"></a>输入&amp;输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p93532063519"><a name="p93532063519"></a><a name="p93532063519"></a>模型推理的输入数据的指针。</p>
<p id="p4388859358"><a name="p4388859358"></a><a name="p4388859358"></a>使用aclmdlDataset类型的数据描述模型推理时的输入数据，输入的内存地址、内存大小用<span id="ph44438521387"><a name="ph44438521387"></a><a name="ph44438521387"></a>aclDataBuffer</span>类型的数据来描述。</p>
</td>
</tr>
<tr id="row147732496171"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p072371917818"><a name="p072371917818"></a><a name="p072371917818"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p177231319387"><a name="p177231319387"></a><a name="p177231319387"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p17715149113710"><a name="p17715149113710"></a><a name="p17715149113710"></a>标识动态维度的输入index。</p>
<p id="p64351912165911"><a name="p64351912165911"></a><a name="p64351912165911"></a>需调用<a href="aclmdlGetInputIndexByName.md">aclmdlGetInputIndexByName</a>接口获取，输入名称固定为ACL_DYNAMIC_TENSOR_NAME。</p>
</td>
</tr>
<tr id="row2604183119586"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p12605123125813"><a name="p12605123125813"></a><a name="p12605123125813"></a>dims</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p18605153110585"><a name="p18605153110585"></a><a name="p18605153110585"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p176141119123511"><a name="p176141119123511"></a><a name="p176141119123511"></a>具体某一档上的所有维度信息的指针。</p>
<p id="p1871113225319"><a name="p1871113225319"></a><a name="p1871113225319"></a>此处设置的动态维度的值只能是模型构建时设置的档位中的某一档。</p>
<p id="p8216277380"><a name="p8216277380"></a><a name="p8216277380"></a><strong id="b850011255407"><a name="b850011255407"></a><a name="b850011255407"></a>例如：</strong>使用ATC工具进行模型转换时，input_shape="data:1,1,40,-1;label:1,-1;mask:-1,-1" ，dynamic_dims="20,20,1,1; 40,40,2,2; 80,60,4,4"，若输入数据的真实维度为（1,1,40,20,1,20,1,1），则dims结构体信息的填充示例如下（name暂不使用）：</p>
<p id="p3774030143811"><a name="p3774030143811"></a><a name="p3774030143811"></a>dims.dimCount = 8</p>
<p id="p16774183093818"><a name="p16774183093818"></a><a name="p16774183093818"></a>dims.dims[0] = 1</p>
<p id="p5774330203820"><a name="p5774330203820"></a><a name="p5774330203820"></a>dims.dims[1] = 1</p>
<p id="p12774143033814"><a name="p12774143033814"></a><a name="p12774143033814"></a>dims.dims[2] = 40</p>
<p id="p677416309385"><a name="p677416309385"></a><a name="p677416309385"></a>dims.dims[3] = 20</p>
<p id="p8419122102812"><a name="p8419122102812"></a><a name="p8419122102812"></a>dims.dims[4] = 1</p>
<p id="p1041918210289"><a name="p1041918210289"></a><a name="p1041918210289"></a>dims.dims[5] = 20</p>
<p id="p204191420283"><a name="p204191420283"></a><a name="p204191420283"></a>dims.dims[6] = 1</p>
<p id="p184191527281"><a name="p184191527281"></a><a name="p184191527281"></a>dims.dims[7] = 1</p>
<p id="p957141754019"><a name="p957141754019"></a><a name="p957141754019"></a>如果不清楚模型构建时的动态维度档位，也可以调用<a href="aclmdlGetInputDynamicDims.md">aclmdlGetInputDynamicDims</a>接口获取指定模型支持的动态维度档位数以及每一档中的值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section7216174405014"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

