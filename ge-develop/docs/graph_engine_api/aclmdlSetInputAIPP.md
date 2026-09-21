# aclmdlSetInputAIPP<a name="ZH-CN_TOPIC_0000001312400601"></a>

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

## 功能说明<a name="section259105813316"></a>

动态AIPP场景下，根据指定的动态AIPP输入的输入index，设置模型推理时的AIPP参数值。

动态AIPP支持的几种操作的计算方式及其计算顺序如下：抠图-\>色域转换-\>缩放（当前版本不支持缩放）-\>减均值/归一化-\>padding。

## 函数原型<a name="section2067518173415"></a>

```
aclError aclmdlSetInputAIPP(uint32_t modelId, aclmdlDataset *dataset, size_t index, const aclmdlAIPP *aippParmsSet)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1919192774810"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p27151254145011"><a name="p27151254145011"></a><a name="p27151254145011"></a>modelId</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1092017278485"><a name="p1092017278485"></a><a name="p1092017278485"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4388859358"><a name="p4388859358"></a><a name="p4388859358"></a>模型的ID。</p>
<p id="p57291851112517"><a name="p57291851112517"></a><a name="p57291851112517"></a>调用<a href="aclmdlLoadFromFile.md">aclmdlLoadFromFile</a>接口/<a href="aclmdlLoadFromMem.md">aclmdlLoadFromMem</a>接口/<a href="aclmdlLoadFromFileWithMem.md">aclmdlLoadFromFileWithMem</a>接口/<a href="aclmdlLoadFromMemWithMem.md">aclmdlLoadFromMemWithMem</a>接口加载模型成功后，会返回模型ID。</p>
</td>
</tr>
<tr id="row132935197287"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p19341911481"><a name="p19341911481"></a><a name="p19341911481"></a>dataset</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p69341916814"><a name="p69341916814"></a><a name="p69341916814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p93532063519"><a name="p93532063519"></a><a name="p93532063519"></a>模型推理的输入数据的指针。</p>
<p id="p1130592352817"><a name="p1130592352817"></a><a name="p1130592352817"></a>使用aclmdlDataset类型的数据描述模型推理时的输入数据，输入的内存地址、内存大小用<span id="ph44438521387"><a name="ph44438521387"></a><a name="ph44438521387"></a>aclDataBuffer</span>类型的数据来描述。</p>
</td>
</tr>
<tr id="row4747721152811"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p072371917818"><a name="p072371917818"></a><a name="p072371917818"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p177231319387"><a name="p177231319387"></a><a name="p177231319387"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p15980142710312"><a name="p15980142710312"></a><a name="p15980142710312"></a>标识动态AIPP输入的输入index。</p>
<p id="p1464223515324"><a name="p1464223515324"></a><a name="p1464223515324"></a>多个动态AIPP输入的场景下，用户可调用<a href="aclmdlGetAippType.md">aclmdlGetAippType</a>接口获取指定模型输入所关联的动态AIPP输入的输入index。</p>
<p id="p64351912165911"><a name="p64351912165911"></a><a name="p64351912165911"></a>为保证向前兼容，如果明确只有一个动态AIPP输入，可调用<a href="aclmdlGetInputIndexByName.md">aclmdlGetInputIndexByName</a>接口获取，输入名称固定为ACL_DYNAMIC_AIPP_NAME。</p>
</td>
</tr>
<tr id="row273214316518"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>aippParmsSet</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p260514111135"><a name="p260514111135"></a><a name="p260514111135"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p956653619549"><a name="p956653619549"></a><a name="p956653619549"></a>动态AIPP参数对象的指针。</p>
<p id="p19724104413549"><a name="p19724104413549"></a><a name="p19724104413549"></a>提前调用<a href="aclmdlCreateAIPP.md">aclmdlCreateAIPP</a>接口创建aclmdlAIPP类型的数据。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section15770391345"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section12107028184916"></a>

-   经过动态AIPP处理后的图像的宽、高必须与原始模型中输入Shape中的宽、高保持一致。
-   多Batch场景下，根据每个Batch的配置计算出动态AIPP后输出图片的宽、高，经过动态AIPP后每个Batch的输出图片宽、高必须是一致的。计算输出图片宽、高的计算公式如[表1](#table81071828124918)所示。
-   抠图或者缩放或者padding之后，对图片宽、高的校验规则如下，其中，aippOutputW、aippOutputH分别表示AIPP输出图片的宽、高，其它参数是[aclmdlSetAIPPSrcImageSize](aclmdlSetAIPPSrcImageSize.md)、[aclmdlSetAIPPScfParams](aclmdlSetAIPPScfParams.md)、[aclmdlSetAIPPCropParams](aclmdlSetAIPPCropParams.md)、[aclmdlSetAIPPPaddingParams](aclmdlSetAIPPPaddingParams.md)接口的入参：

    **表 1**  输出图片宽、高计算公式

    <a name="table81071828124918"></a>
    <table><thead align="left"><tr id="row1010717284494"><th class="cellrowborder" valign="top" width="22.277772222777724%" id="mcps1.2.5.1.1"><p id="p210772894913"><a name="p210772894913"></a><a name="p210772894913"></a>抠图</p>
    </th>
    <th class="cellrowborder" valign="top" width="19.468053194680536%" id="mcps1.2.5.1.2"><p id="p1410714281497"><a name="p1410714281497"></a><a name="p1410714281497"></a>缩放</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.1982801719828%" id="mcps1.2.5.1.3"><p id="p310732874910"><a name="p310732874910"></a><a name="p310732874910"></a>补边（padding）</p>
    </th>
    <th class="cellrowborder" valign="top" width="41.05589441055895%" id="mcps1.2.5.1.4"><p id="p151078283496"><a name="p151078283496"></a><a name="p151078283496"></a>动态AIPP输出图片的宽、高</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row410711287497"><td class="cellrowborder" valign="top" width="22.277772222777724%" headers="mcps1.2.5.1.1 "><p id="p1110717282498"><a name="p1110717282498"></a><a name="p1110717282498"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.468053194680536%" headers="mcps1.2.5.1.2 "><p id="p0108728124911"><a name="p0108728124911"></a><a name="p0108728124911"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.1982801719828%" headers="mcps1.2.5.1.3 "><p id="p1110818284492"><a name="p1110818284492"></a><a name="p1110818284492"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="41.05589441055895%" headers="mcps1.2.5.1.4 "><p id="p51085281499"><a name="p51085281499"></a><a name="p51085281499"></a>aippOutputW=srcImageSizeW，aippOutputH=srcImageSizeH</p>
    </td>
    </tr>
    <tr id="row191081285491"><td class="cellrowborder" valign="top" width="22.277772222777724%" headers="mcps1.2.5.1.1 "><p id="p12108728184920"><a name="p12108728184920"></a><a name="p12108728184920"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.468053194680536%" headers="mcps1.2.5.1.2 "><p id="p9108128154917"><a name="p9108128154917"></a><a name="p9108128154917"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.1982801719828%" headers="mcps1.2.5.1.3 "><p id="p18108172812494"><a name="p18108172812494"></a><a name="p18108172812494"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="41.05589441055895%" headers="mcps1.2.5.1.4 "><p id="p15108152854916"><a name="p15108152854916"></a><a name="p15108152854916"></a>aippOutputW=cropSizeW，aippOutputH=cropSizeH</p>
    </td>
    </tr>
    <tr id="row11081928174913"><td class="cellrowborder" valign="top" width="22.277772222777724%" headers="mcps1.2.5.1.1 "><p id="p8108142815491"><a name="p8108142815491"></a><a name="p8108142815491"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.468053194680536%" headers="mcps1.2.5.1.2 "><p id="p1710820281490"><a name="p1710820281490"></a><a name="p1710820281490"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.1982801719828%" headers="mcps1.2.5.1.3 "><p id="p151086288498"><a name="p151086288498"></a><a name="p151086288498"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="41.05589441055895%" headers="mcps1.2.5.1.4 "><p id="p7108152812492"><a name="p7108152812492"></a><a name="p7108152812492"></a>aippOutputW=scfOutputSizeW，aippOutputH=scfOutputSizeH</p>
    </td>
    </tr>
    <tr id="row151084288499"><td class="cellrowborder" valign="top" width="22.277772222777724%" headers="mcps1.2.5.1.1 "><p id="p210810285498"><a name="p210810285498"></a><a name="p210810285498"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.468053194680536%" headers="mcps1.2.5.1.2 "><p id="p4108192854917"><a name="p4108192854917"></a><a name="p4108192854917"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.1982801719828%" headers="mcps1.2.5.1.3 "><p id="p810852813492"><a name="p810852813492"></a><a name="p810852813492"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="41.05589441055895%" headers="mcps1.2.5.1.4 "><p id="p810818284494"><a name="p810818284494"></a><a name="p810818284494"></a>aippOutputW=cropSizeW + paddingSizeLeft + paddingSizeRight，aippOutputH=cropSizeH + paddingSizeTop + paddingSizeBottom</p>
    </td>
    </tr>
    <tr id="row71080282493"><td class="cellrowborder" valign="top" width="22.277772222777724%" headers="mcps1.2.5.1.1 "><p id="p4108172818492"><a name="p4108172818492"></a><a name="p4108172818492"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.468053194680536%" headers="mcps1.2.5.1.2 "><p id="p1610811286494"><a name="p1610811286494"></a><a name="p1610811286494"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.1982801719828%" headers="mcps1.2.5.1.3 "><p id="p1110842812491"><a name="p1110842812491"></a><a name="p1110842812491"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="41.05589441055895%" headers="mcps1.2.5.1.4 "><p id="p0108142814490"><a name="p0108142814490"></a><a name="p0108142814490"></a>aippOutputW=srcImageSizeW + paddingSizeLeft + paddingSizeRight，aippOutputH=srcImageSizeH + paddingSizeTop + paddingSizeBottom</p>
    </td>
    </tr>
    <tr id="row10108328114920"><td class="cellrowborder" valign="top" width="22.277772222777724%" headers="mcps1.2.5.1.1 "><p id="p2108162854914"><a name="p2108162854914"></a><a name="p2108162854914"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.468053194680536%" headers="mcps1.2.5.1.2 "><p id="p110914286495"><a name="p110914286495"></a><a name="p110914286495"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.1982801719828%" headers="mcps1.2.5.1.3 "><p id="p13109182814495"><a name="p13109182814495"></a><a name="p13109182814495"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="41.05589441055895%" headers="mcps1.2.5.1.4 "><p id="p1510915285498"><a name="p1510915285498"></a><a name="p1510915285498"></a>aippOutputW=scfOutputSizeW + paddingSizeLeft + paddingSizeRight，aippOutputH=scfOutputSizeH + paddingSizeTop + paddingSizeBottom</p>
    </td>
    </tr>
    <tr id="row910962824919"><td class="cellrowborder" valign="top" width="22.277772222777724%" headers="mcps1.2.5.1.1 "><p id="p210992813499"><a name="p210992813499"></a><a name="p210992813499"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.468053194680536%" headers="mcps1.2.5.1.2 "><p id="p7109132812496"><a name="p7109132812496"></a><a name="p7109132812496"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.1982801719828%" headers="mcps1.2.5.1.3 "><p id="p14109122874918"><a name="p14109122874918"></a><a name="p14109122874918"></a>否</p>
    </td>
    <td class="cellrowborder" valign="top" width="41.05589441055895%" headers="mcps1.2.5.1.4 "><p id="p3109102818492"><a name="p3109102818492"></a><a name="p3109102818492"></a>aippOutputW=scfOutputSizeW，aippOutputH=scfOutputSizeH</p>
    </td>
    </tr>
    <tr id="row910912819495"><td class="cellrowborder" valign="top" width="22.277772222777724%" headers="mcps1.2.5.1.1 "><p id="p101091228164914"><a name="p101091228164914"></a><a name="p101091228164914"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.468053194680536%" headers="mcps1.2.5.1.2 "><p id="p18109928104917"><a name="p18109928104917"></a><a name="p18109928104917"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.1982801719828%" headers="mcps1.2.5.1.3 "><p id="p13109112812492"><a name="p13109112812492"></a><a name="p13109112812492"></a>是</p>
    </td>
    <td class="cellrowborder" valign="top" width="41.05589441055895%" headers="mcps1.2.5.1.4 "><p id="p12109162810496"><a name="p12109162810496"></a><a name="p12109162810496"></a>aippOutputW=scfOutputSizeW + paddingSizeLeft + paddingSizeRight，aippOutputH=scfOutputSizeH + paddingSizeTop + paddingSizeBottom</p>
    </td>
    </tr>
    </tbody>
    </table>

