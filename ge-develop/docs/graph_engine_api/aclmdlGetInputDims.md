# aclmdlGetInputDims<a name="ZH-CN_TOPIC_0000001312641477"></a>

## 产品支持情况<a name="section16107182283615"></a>

<a name="zh-cn_topic_0000002219420921_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002219420921_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002219420921_p1883113061818"><a name="zh-cn_topic_0000002219420921_p1883113061818"></a><a name="zh-cn_topic_0000002219420921_p1883113061818"></a><span id="zh-cn_topic_0000002219420921_ph20833205312295"><a name="zh-cn_topic_0000002219420921_ph20833205312295"></a><a name="zh-cn_topic_0000002219420921_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002219420921_p783113012187"><a name="zh-cn_topic_0000002219420921_p783113012187"></a><a name="zh-cn_topic_0000002219420921_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002219420921_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p48327011813"><a name="zh-cn_topic_0000002219420921_p48327011813"></a><a name="zh-cn_topic_0000002219420921_p48327011813"></a><span id="zh-cn_topic_0000002219420921_ph583230201815"><a name="zh-cn_topic_0000002219420921_ph583230201815"></a><a name="zh-cn_topic_0000002219420921_ph583230201815"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p7948163910184"><a name="zh-cn_topic_0000002219420921_p7948163910184"></a><a name="zh-cn_topic_0000002219420921_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002219420921_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p14832120181815"><a name="zh-cn_topic_0000002219420921_p14832120181815"></a><a name="zh-cn_topic_0000002219420921_p14832120181815"></a><span id="zh-cn_topic_0000002219420921_ph1483216010188"><a name="zh-cn_topic_0000002219420921_ph1483216010188"></a><a name="zh-cn_topic_0000002219420921_ph1483216010188"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p19948143911820"><a name="zh-cn_topic_0000002219420921_p19948143911820"></a><a name="zh-cn_topic_0000002219420921_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section187304428437"></a>

根据模型描述信息获取模型的输入tensor的维度信息。

如果模型中含有静态AIPP配置信息，您可以根据实际需要选择[aclmdlGetInputDims](aclmdlGetInputDims.md)接口或[aclmdlGetInputDimsV2](aclmdlGetInputDimsV2.md)接口查询维度信息，两者的区别在于：

-   通过[aclmdlGetInputDims](aclmdlGetInputDims.md)接口获取的维度信息，各维度的值与输入图像的各维度的值保持一致，详细规则如[表1](#table593133825712)所示。
-   通过[aclmdlGetInputDimsV2](aclmdlGetInputDimsV2.md)接口获取的维度信息，H维度的值 = 输入图像的H维度值\*系数，不同图片格式下C维度的值不同，详细规则如[aclmdlGetInputDimsV2](aclmdlGetInputDimsV2.md)接口处的[表1](aclmdlGetInputDimsV2.md#table593133825712)所示，且各维度值相乘的结果值与通过[aclmdlGetInputSizeByIndex](aclmdlGetInputSizeByIndex.md)接口获取的值保持一致。

## 函数原型<a name="section18474205015436"></a>

```
aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
```

## 参数说明<a name="section135141717811"></a>

<a name="table1831221142"></a>
<table><thead align="left"><tr id="row3332181419"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p11372191420"><a name="p11372191420"></a><a name="p11372191420"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p13320291417"><a name="p13320291417"></a><a name="p13320291417"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p1940213149"><a name="p1940213149"></a><a name="p1940213149"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row134122191416"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p14132111415"><a name="p14132111415"></a><a name="p14132111415"></a>modelDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p144122141415"><a name="p144122141415"></a><a name="p144122141415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4414215143"><a name="p4414215143"></a><a name="p4414215143"></a>aclmdlDesc类型的指针。</p>
<p id="p35754134454"><a name="p35754134454"></a><a name="p35754134454"></a>需提前调用<a href="aclmdlCreateDesc.md">aclmdlCreateDesc</a>接口创建aclmdlDesc类型的数据。</p>
</td>
</tr>
<tr id="row218712903311"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p8275142112445"><a name="p8275142112445"></a><a name="p8275142112445"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p14274152194413"><a name="p14274152194413"></a><a name="p14274152194413"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4273172144416"><a name="p4273172144416"></a><a name="p4273172144416"></a>指定获取第几个输入的Dims，index值从0开始。</p>
</td>
</tr>
<tr id="row1391733115319"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p204083318531"><a name="p204083318531"></a><a name="p204083318531"></a>dims</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p184012339539"><a name="p184012339539"></a><a name="p184012339539"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p8128113683010"><a name="p8128113683010"></a><a name="p8128113683010"></a>输入维度信息的指针。</p>
<p id="p14690189135514"><a name="p14690189135514"></a><a name="p14690189135514"></a>针对动态Batch、动态分辨率（宽高）的场景，输入tensor的dims中batch size或宽高为-1，表示其动态可变。例如，输入tensor的format为NCHW，在动态Batch场景下，动态可变的输入tensor的dims为[<strong id="b7706143419135"><a name="b7706143419135"></a><a name="b7706143419135"></a>-1</strong>,<em id="i1267910510145"><a name="i1267910510145"></a><a name="i1267910510145"></a>3</em>,<em id="i1866155621419"><a name="i1866155621419"></a><a name="i1866155621419"></a>224</em>,<em id="i1748295912140"><a name="i1748295912140"></a><a name="i1748295912140"></a>224</em>]；在动态分辨率场景下，动态可变的输入tensor的dims为[<em id="i167587321516"><a name="i167587321516"></a><a name="i167587321516"></a>1</em>,<em id="i1783714810158"><a name="i1783714810158"></a><a name="i1783714810158"></a>3</em>,<strong id="b10441171511147"><a name="b10441171511147"></a><a name="b10441171511147"></a>-1</strong>,<strong id="b1999162181411"><a name="b1999162181411"></a><a name="b1999162181411"></a>-1</strong>]。举例中的斜体部分以实际情况为准。</p>
<p id="p149381612192019"><a name="p149381612192019"></a><a name="p149381612192019"></a>若tensor的name长度大于127，则在输出的dims.name时，接口会将tensor的name转换为“acl_modelId_<em id="i678714213346"><a name="i678714213346"></a><a name="i678714213346"></a>${id}</em>_input_<em id="i4983860345"><a name="i4983860345"></a><a name="i4983860345"></a>${index}</em>_<em id="i12443055216"><a name="i12443055216"></a><a name="i12443055216"></a>${随机字符串}</em>  ”格式（如果转换后的tensor的name与模型中已有的tensor的name冲突，则会在转换后的name尾部增加“_<em id="i1388175614276"><a name="i1388175614276"></a><a name="i1388175614276"></a>${随机字符串}</em> ”，否则不会增加随机字符串），并在转换后的name与原name之间建立映射关系，用户可调用<a href="aclmdlGetTensorRealName.md">aclmdlGetTensorRealName</a>接口，传入转换后的name，获取原name（若向接口传入原name，则获取的还是原name）；若tensor的name长度小于或等于127，则在输出的dims.name时，按tensor的name输出。</p>
<p id="p937412298552"><a name="p937412298552"></a><a name="p937412298552"></a>针对静态AIPP场景，本接口针对不同格式的图像，对应NHWC的Format格式，当前接口中明确各个维度的定义规则，如<a href="#table593133825712">表1</a>所示。</p>
</td>
</tr>
</tbody>
</table>

**表 1**  静态AIPP场景下的维度定义规则

<a name="table593133825712"></a>
<table><thead align="left"><tr id="row4973838175716"><th class="cellrowborder" valign="top" width="32.26322632263226%" id="mcps1.2.4.1.1"><p id="p109731938125715"><a name="p109731938125715"></a><a name="p109731938125715"></a>图像格式</p>
</th>
<th class="cellrowborder" valign="top" width="34.4034403440344%" id="mcps1.2.4.1.2"><p id="p1397383814577"><a name="p1397383814577"></a><a name="p1397383814577"></a>Format参考格式</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.3"><p id="p89731638115711"><a name="p89731638115711"></a><a name="p89731638115711"></a>维度定义规则</p>
</th>
</tr>
</thead>
<tbody><tr id="row6973143814579"><td class="cellrowborder" valign="top" width="32.26322632263226%" headers="mcps1.2.4.1.1 "><p id="p597312387578"><a name="p597312387578"></a><a name="p597312387578"></a>YUV420SP_U8</p>
</td>
<td class="cellrowborder" valign="top" width="34.4034403440344%" headers="mcps1.2.4.1.2 "><p id="p897303814573"><a name="p897303814573"></a><a name="p897303814573"></a>NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p1097383814571"><a name="p1097383814571"></a><a name="p1097383814571"></a>n,h,w,c</p>
</td>
</tr>
<tr id="row3973338155720"><td class="cellrowborder" valign="top" width="32.26322632263226%" headers="mcps1.2.4.1.1 "><p id="p59731738175718"><a name="p59731738175718"></a><a name="p59731738175718"></a>XRGB8888_U8</p>
</td>
<td class="cellrowborder" valign="top" width="34.4034403440344%" headers="mcps1.2.4.1.2 "><p id="p297315388578"><a name="p297315388578"></a><a name="p297315388578"></a>NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p13543172119011"><a name="p13543172119011"></a><a name="p13543172119011"></a>n,h,w,c</p>
</td>
</tr>
<tr id="row29731238165714"><td class="cellrowborder" valign="top" width="32.26322632263226%" headers="mcps1.2.4.1.1 "><p id="p697303817577"><a name="p697303817577"></a><a name="p697303817577"></a>RGB888_U8</p>
</td>
<td class="cellrowborder" valign="top" width="34.4034403440344%" headers="mcps1.2.4.1.2 "><p id="p11973538155714"><a name="p11973538155714"></a><a name="p11973538155714"></a>NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p813414244019"><a name="p813414244019"></a><a name="p813414244019"></a>n,h,w,c</p>
</td>
</tr>
<tr id="row6973938145711"><td class="cellrowborder" valign="top" width="32.26322632263226%" headers="mcps1.2.4.1.1 "><p id="p1697493815711"><a name="p1697493815711"></a><a name="p1697493815711"></a>YUV400_U8</p>
</td>
<td class="cellrowborder" valign="top" width="34.4034403440344%" headers="mcps1.2.4.1.2 "><p id="p1397453825710"><a name="p1397453825710"></a><a name="p1397453825710"></a>NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p96214261401"><a name="p96214261401"></a><a name="p96214261401"></a>n,h,w,c</p>
</td>
</tr>
<tr id="row159741387578"><td class="cellrowborder" valign="top" width="32.26322632263226%" headers="mcps1.2.4.1.1 "><p id="p11974123895717"><a name="p11974123895717"></a><a name="p11974123895717"></a>ARGB8888_U8</p>
</td>
<td class="cellrowborder" valign="top" width="34.4034403440344%" headers="mcps1.2.4.1.2 "><p id="p1397483805713"><a name="p1397483805713"></a><a name="p1397483805713"></a>NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p613752914019"><a name="p613752914019"></a><a name="p613752914019"></a>n,h,w,c</p>
</td>
</tr>
<tr id="row149741538145719"><td class="cellrowborder" valign="top" width="32.26322632263226%" headers="mcps1.2.4.1.1 "><p id="p109744385571"><a name="p109744385571"></a><a name="p109744385571"></a>YUYV_U8</p>
</td>
<td class="cellrowborder" valign="top" width="34.4034403440344%" headers="mcps1.2.4.1.2 "><p id="p797416389570"><a name="p797416389570"></a><a name="p797416389570"></a>NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p1035383119014"><a name="p1035383119014"></a><a name="p1035383119014"></a>n,h,w,c</p>
</td>
</tr>
<tr id="row89741738185713"><td class="cellrowborder" valign="top" width="32.26322632263226%" headers="mcps1.2.4.1.1 "><p id="p0974113819579"><a name="p0974113819579"></a><a name="p0974113819579"></a>YUV422SP_U8</p>
</td>
<td class="cellrowborder" valign="top" width="34.4034403440344%" headers="mcps1.2.4.1.2 "><p id="p59741638105710"><a name="p59741638105710"></a><a name="p59741638105710"></a>NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p2060219331018"><a name="p2060219331018"></a><a name="p2060219331018"></a>n,h,w,c</p>
</td>
</tr>
<tr id="row197433875714"><td class="cellrowborder" valign="top" width="32.26322632263226%" headers="mcps1.2.4.1.1 "><p id="p5974103814576"><a name="p5974103814576"></a><a name="p5974103814576"></a>AYUV444_U8</p>
</td>
<td class="cellrowborder" valign="top" width="34.4034403440344%" headers="mcps1.2.4.1.2 "><p id="p1097443817571"><a name="p1097443817571"></a><a name="p1097443817571"></a>NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p18569193613019"><a name="p18569193613019"></a><a name="p18569193613019"></a>n,h,w,c</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section184151120582"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

