# aclmdlSetAIPPCscParams<a name="ZH-CN_TOPIC_0000001312721781"></a>

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

## 功能说明<a name="section1857616014371"></a>

动态AIPP场景下，设置CSC色域转换相关的参数，若色域转换开关关闭，则调用该接口设置以下参数无效。

```
YUV转BGR：
| B |   | cscMatrixR0C0 cscMatrixR0C1 cscMatrixR0C2 | | Y - cscInputBiasR0 |
| G | = | cscMatrixR1C0 cscMatrixR1C1 cscMatrixR1C2 | | U - cscInputBiasR1 | >> 8
| R |   | cscMatrixR2C0 cscMatrixR2C1 cscMatrixR2C2 | | V - cscInputBiasR2 |
BGR转YUV：
| Y |   | cscMatrixR0C0 cscMatrixR0C1 cscMatrixR0C2 | | B |        | cscOutputBiasR0 |
| U | = | cscMatrixR1C0 cscMatrixR1C1 cscMatrixR1C2 | | G | >> 8 + | cscOutputBiasR1 |
| V |   | cscMatrixR2C0 cscMatrixR2C1 cscMatrixR2C2 | | R |        | cscOutputBiasR2 |
```

色域转换参数值与转换前图片的格式、转换后图片的格式强相关，您可以参考《ATC离线模型编译工具用户指南》，根据转换前图片格式、转换后图片格式来配置色域转换参数。如果手册中列举的图片格式不满足要求，您需自行根据实际需求配置色域转换参数。

## 函数原型<a name="section18417410375"></a>

```
aclError aclmdlSetAIPPCscParams(aclmdlAIPP *aippParmsSet, int8_t cscSwitch,
int16_t cscMatrixR0C0, int16_t cscMatrixR0C1, int16_t cscMatrixR0C2,
int16_t cscMatrixR1C0, int16_t cscMatrixR1C1,int16_t cscMatrixR1C2,
int16_t cscMatrixR2C0, int16_t cscMatrixR2C1, int16_t cscMatrixR2C2,
uint8_t cscOutputBiasR0, uint8_t cscOutputBiasR1, uint8_t cscOutputBiasR2,
uint8_t cscInputBiasR0, uint8_t cscInputBiasR1, uint8_t cscInputBiasR2)
```

## 参数说明<a name="section1162911716377"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p039116593511"><a name="p039116593511"></a><a name="p039116593511"></a>aippParmsSet</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p260514111135"><a name="p260514111135"></a><a name="p260514111135"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p956653619549"><a name="p956653619549"></a><a name="p956653619549"></a>动态AIPP参数对象的指针。</p>
<p id="p19724104413549"><a name="p19724104413549"></a><a name="p19724104413549"></a>需提前调用<a href="aclmdlCreateAIPP.md">aclmdlCreateAIPP</a>接口创建aclmdlAIPP类型的数据。</p>
</td>
</tr>
<tr id="row7909131293411"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p25975757"><a name="p25975757"></a><a name="p25975757"></a>cscSwitch</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p69341916814"><a name="p69341916814"></a><a name="p69341916814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p156501849144218"><a name="p156501849144218"></a><a name="p156501849144218"></a>色域转换开关，取值范围：</p>
<a name="ul12256145194212"></a><a name="ul12256145194212"></a><ul id="ul12256145194212"><li>0：关闭色域转换开关，设置为0时，则设置以下参数无效</li><li>1：打开色域转换开关</li></ul>
<p id="p20682694"><a name="p20682694"></a><a name="p20682694"></a></p>
</td>
</tr>
<tr id="row1866142612125"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p45298596"><a name="p45298596"></a><a name="p45298596"></a>cscMatrixR0C0</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p9672026141215"><a name="p9672026141215"></a><a name="p9672026141215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p103111627104016"><a name="p103111627104016"></a><a name="p103111627104016"></a>色域转换矩阵参数。</p>
<p id="p6217718417"><a name="p6217718417"></a><a name="p6217718417"></a>取值范围：[-32677 ,32676]</p>
<p id="p38378011"><a name="p38378011"></a><a name="p38378011"></a></p>
</td>
</tr>
<tr id="row1638310288123"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p60282721"><a name="p60282721"></a><a name="p60282721"></a>cscMatrixR0C1</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p43831128141211"><a name="p43831128141211"></a><a name="p43831128141211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1111753020409"><a name="p1111753020409"></a><a name="p1111753020409"></a>色域转换矩阵参数。</p>
<p id="p015701111417"><a name="p015701111417"></a><a name="p015701111417"></a>取值范围：[-32677 ,32676]</p>
<p id="p11982196"><a name="p11982196"></a><a name="p11982196"></a></p>
</td>
</tr>
<tr id="row6740123014128"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p10868702"><a name="p10868702"></a><a name="p10868702"></a>cscMatrixR0C2</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p117401830101219"><a name="p117401830101219"></a><a name="p117401830101219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p45441431104010"><a name="p45441431104010"></a><a name="p45441431104010"></a>色域转换矩阵参数。</p>
<p id="p1746910133418"><a name="p1746910133418"></a><a name="p1746910133418"></a>取值范围：[-32677 ,32676]</p>
<p id="p13994266"><a name="p13994266"></a><a name="p13994266"></a></p>
</td>
</tr>
<tr id="row1711203311124"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1272826"><a name="p1272826"></a><a name="p1272826"></a>cscMatrixR1C0</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p5711173351210"><a name="p5711173351210"></a><a name="p5711173351210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1247123334010"><a name="p1247123334010"></a><a name="p1247123334010"></a>色域转换矩阵参数。</p>
<p id="p1431771718412"><a name="p1431771718412"></a><a name="p1431771718412"></a>取值范围：[-32677 ,32676]</p>
<p id="p42015198"><a name="p42015198"></a><a name="p42015198"></a></p>
</td>
</tr>
<tr id="row143961136111218"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p27437792"><a name="p27437792"></a><a name="p27437792"></a>cscMatrixR1C1</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p33961836121216"><a name="p33961836121216"></a><a name="p33961836121216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p84601334124013"><a name="p84601334124013"></a><a name="p84601334124013"></a>色域转换矩阵参数。</p>
<p id="p1163791994113"><a name="p1163791994113"></a><a name="p1163791994113"></a>取值范围：[-32677 ,32676]</p>
<p id="p19572825"><a name="p19572825"></a><a name="p19572825"></a></p>
</td>
</tr>
<tr id="row738013861219"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p41510976"><a name="p41510976"></a><a name="p41510976"></a>cscMatrixR1C2</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1738093831210"><a name="p1738093831210"></a><a name="p1738093831210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p17533103754013"><a name="p17533103754013"></a><a name="p17533103754013"></a>色域转换矩阵参数。</p>
<p id="p17101422104116"><a name="p17101422104116"></a><a name="p17101422104116"></a>取值范围：[-32677 ,32676]</p>
<p id="p4860205"><a name="p4860205"></a><a name="p4860205"></a></p>
</td>
</tr>
<tr id="row48561340121212"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p53428882"><a name="p53428882"></a><a name="p53428882"></a>cscMatrixR2C0</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1185614019129"><a name="p1185614019129"></a><a name="p1185614019129"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p2227173918401"><a name="p2227173918401"></a><a name="p2227173918401"></a>色域转换矩阵参数。</p>
<p id="p7500202484115"><a name="p7500202484115"></a><a name="p7500202484115"></a>取值范围：[-32677 ,32676]</p>
<p id="p1740610"><a name="p1740610"></a><a name="p1740610"></a></p>
</td>
</tr>
<tr id="row392144241215"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p60945241"><a name="p60945241"></a><a name="p60945241"></a>cscMatrixR2C1</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p8921114251210"><a name="p8921114251210"></a><a name="p8921114251210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p513204134014"><a name="p513204134014"></a><a name="p513204134014"></a>色域转换矩阵参数。</p>
<p id="p19861122674114"><a name="p19861122674114"></a><a name="p19861122674114"></a>取值范围：[-32677 ,32676]</p>
<p id="p49289224"><a name="p49289224"></a><a name="p49289224"></a></p>
</td>
</tr>
<tr id="row38959444122"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p28602718"><a name="p28602718"></a><a name="p28602718"></a>cscMatrixR2C2</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p138955440123"><a name="p138955440123"></a><a name="p138955440123"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p966904210405"><a name="p966904210405"></a><a name="p966904210405"></a>色域转换矩阵参数。</p>
<p id="p585152934114"><a name="p585152934114"></a><a name="p585152934114"></a>取值范围：[-32677 ,32676]</p>
<p id="p29641146"><a name="p29641146"></a><a name="p29641146"></a></p>
</td>
</tr>
<tr id="row69964710128"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p66450672"><a name="p66450672"></a><a name="p66450672"></a>cscOutputBiasR0</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p79913475129"><a name="p79913475129"></a><a name="p79913475129"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p17628111044220"><a name="p17628111044220"></a><a name="p17628111044220"></a>RGB转YUV时的输出偏移。</p>
<p id="p12460147164217"><a name="p12460147164217"></a><a name="p12460147164217"></a>取值范围：[0, 255]</p>
<p id="p48477080"><a name="p48477080"></a><a name="p48477080"></a></p>
</td>
</tr>
<tr id="row14396174913121"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p40529449"><a name="p40529449"></a><a name="p40529449"></a>cscOutputBiasR1</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1039616493127"><a name="p1039616493127"></a><a name="p1039616493127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10380191284213"><a name="p10380191284213"></a><a name="p10380191284213"></a>RGB转YUV时的输出偏移。</p>
<p id="p0931950204217"><a name="p0931950204217"></a><a name="p0931950204217"></a>取值范围：[0, 255]</p>
<p id="p18743339"><a name="p18743339"></a><a name="p18743339"></a></p>
</td>
</tr>
<tr id="row2718125116129"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p40795435"><a name="p40795435"></a><a name="p40795435"></a>cscOutputBiasR2</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p127181651111218"><a name="p127181651111218"></a><a name="p127181651111218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p180516131427"><a name="p180516131427"></a><a name="p180516131427"></a>RGB转YUV时的输出偏移。</p>
<p id="p1931055244216"><a name="p1931055244216"></a><a name="p1931055244216"></a>取值范围：[0, 255]</p>
<p id="p43084165"><a name="p43084165"></a><a name="p43084165"></a></p>
</td>
</tr>
<tr id="row4198125416128"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1408093"><a name="p1408093"></a><a name="p1408093"></a>cscInputBiasR0</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p01987543129"><a name="p01987543129"></a><a name="p01987543129"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p186001554216"><a name="p186001554216"></a><a name="p186001554216"></a>YUV转RGB时的输入偏移。</p>
<p id="p113151054194219"><a name="p113151054194219"></a><a name="p113151054194219"></a>取值范围：[0, 255]</p>
<p id="p54992765"><a name="p54992765"></a><a name="p54992765"></a></p>
</td>
</tr>
<tr id="row1171835881217"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p25734322"><a name="p25734322"></a><a name="p25734322"></a>cscInputBiasR1</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1471865851214"><a name="p1471865851214"></a><a name="p1471865851214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p61674174424"><a name="p61674174424"></a><a name="p61674174424"></a>YUV转RGB时的输入偏移。</p>
<p id="p0606195614422"><a name="p0606195614422"></a><a name="p0606195614422"></a>取值范围：[0, 255]</p>
<p id="p24555510"><a name="p24555510"></a><a name="p24555510"></a></p>
</td>
</tr>
<tr id="row1817133014136"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p517113304139"><a name="p517113304139"></a><a name="p517113304139"></a>cscInputBiasR2</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p417123021319"><a name="p417123021319"></a><a name="p417123021319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1286116189429"><a name="p1286116189429"></a><a name="p1286116189429"></a>YUV转RGB时的输入偏移。</p>
<p id="p473317584425"><a name="p473317584425"></a><a name="p473317584425"></a>取值范围：[0, 255]</p>
<p id="p9109783"><a name="p9109783"></a><a name="p9109783"></a></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section16656141073717"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明<a name="section72671611204610"></a>

如果通过[aclmdlSetAIPPInputFormat](aclmdlSetAIPPInputFormat.md)接口设置的原始图像格式为YUV400，则不支持通过本接口设置色域转换参数。

