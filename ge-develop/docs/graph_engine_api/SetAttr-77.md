# SetAttr<a name="ZH-CN_TOPIC_0000002499360476"></a>

## 产品支持情况<a name="section789110355111"></a>

<a name="zh-cn_topic_0000001312404881_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312404881_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001312404881_p1883113061818"><a name="zh-cn_topic_0000001312404881_p1883113061818"></a><a name="zh-cn_topic_0000001312404881_p1883113061818"></a><span id="zh-cn_topic_0000001312404881_ph20833205312295"><a name="zh-cn_topic_0000001312404881_ph20833205312295"></a><a name="zh-cn_topic_0000001312404881_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001312404881_p783113012187"><a name="zh-cn_topic_0000001312404881_p783113012187"></a><a name="zh-cn_topic_0000001312404881_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312404881_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312404881_p48327011813"><a name="zh-cn_topic_0000001312404881_p48327011813"></a><a name="zh-cn_topic_0000001312404881_p48327011813"></a><span id="zh-cn_topic_0000001312404881_ph583230201815"><a name="zh-cn_topic_0000001312404881_ph583230201815"></a><a name="zh-cn_topic_0000001312404881_ph583230201815"></a><term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312404881_p108715341013"><a name="zh-cn_topic_0000001312404881_p108715341013"></a><a name="zh-cn_topic_0000001312404881_p108715341013"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312404881_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312404881_p14832120181815"><a name="zh-cn_topic_0000001312404881_p14832120181815"></a><a name="zh-cn_topic_0000001312404881_p14832120181815"></a><span id="zh-cn_topic_0000001312404881_ph1483216010188"><a name="zh-cn_topic_0000001312404881_ph1483216010188"></a><a name="zh-cn_topic_0000001312404881_ph1483216010188"></a><term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312404881_p19948143911820"><a name="zh-cn_topic_0000001312404881_p19948143911820"></a><a name="zh-cn_topic_0000001312404881_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 头文件和库文件<a name="section094612611340"></a>

-   头文件：\#include <graph/operator.h\>
-   库文件：libgraph.so

## 功能说明<a name="section10161546"></a>

设置算子属性的属性值。

算子可以包括多个属性，初次设置值后，算子属性值的类型固定，算子属性值的类型包括：

-   整型：接受int64\_t、uint32\_t、int32\_t类型的整型值

    使用SetAttr\(const string& name, int64\_t attrValue\)设置属性值，以GetAttr\(const string& name, int32\_t& attrValue\) 、GetAttr\(const string& name, uint32\_t& attrValue\) 取值时，用户需保证整型数据没有截断，同理针对int32\_t和uint32\_t混用时需要保证不被截断。

-   整型列表：接受std::vector<int64\_t\>、std::vector<int32\_t\>、std::vector<uint32\_t\>、std::initializer\_list<int64\_t\>&&表示的整型列表数据
-   浮点数：float
-   浮点数列表：std::vector<float\>
-   字符串：string
-   字符串列表：std::vector<std::string\>
-   布尔：bool
-   布尔列表：std::vector<bool\>
-   Tensor：Tensor
-   Tensor列表：std::vector<Tensor\>
-   Bytes：字节数组，SetAttr接受通过OpBytes（即vector<uint8\_t\>），和（const uint8\_t\* data, size\_t size）表示的字节数组
-   AttrValue类型
-   整型二维列表类型：std::vector<std::vector<int64\_t\>\>
-   DataType列表类型：std::vector<ge::DataType\>
-   DataType类型：ge::DataType
-   NamedAttrs类型： ge::NamedAttrs
-   NamedAttrs列表类型：std::vector<ge::NamedAttrs\>

## 函数原型<a name="section182148481597"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
Operator &SetAttr(const std::string &name, int64_t attr_value)
Operator &SetAttr(const std::string &name, int32_t attr_value)
Operator &SetAttr(const std::string &name, uint32_t attr_value)
Operator &SetAttr(const std::string &name, const std::vector<int64_t> &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<int32_t> &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<uint32_t> &attr_value)
Operator &SetAttr(const std::string &name, std::initializer_list<int64_t> &&attr_value)
Operator &SetAttr(const std::string &name, float32_t attr_value)
Operator &SetAttr(const std::string &name, const std::vector<float32_t> &attr_value)
Operator &SetAttr(const std::string &name, AttrValue &&attr_value)
Operator &SetAttr(const std::string &name, const std::string &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<std::string> &attr_value)
Operator &SetAttr(const std::string &name, bool attr_value)
Operator &SetAttr(const std::string &name, const std::vector<bool> &attr_value)
Operator &SetAttr(const std::string &name, const Tensor &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<Tensor> &attr_value)
Operator &SetAttr(const std::string &name, const OpBytes &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<std::vector<int64_t>> &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<ge::DataType> &attr_value)
Operator &SetAttr(const std::string &name, const ge::DataType &attr_value)
Operator &SetAttr(const std::string &name, const ge::NamedAttrs &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<ge::NamedAttrs> &attr_value)
Operator &SetAttr(const char_t *name, int64_t attr_value)
Operator &SetAttr(const char_t *name, int32_t attr_value)
Operator &SetAttr(const char_t *name, uint32_t attr_value)
Operator &SetAttr(const char_t *name, const std::vector<int64_t> &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<int32_t> &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<uint32_t> &attr_value)
Operator &SetAttr(const char_t *name, std::initializer_list<int64_t> &&attr_value)
Operator &SetAttr(const char_t *name, float32_t attr_value)
Operator &SetAttr(const char_t *name, const std::vector<float32_t> &attr_value)
Operator &SetAttr(const char_t *name, AttrValue &&attr_value)
Operator &SetAttr(const char_t *name, const char_t *attr_value)
Operator &SetAttr(const char_t *name, const AscendString &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<AscendString> &attr_values)
Operator &SetAttr(const char_t *name, bool attr_value)
Operator &SetAttr(const char_t *name, const std::vector<bool> &attr_value)
Operator &SetAttr(const char_t *name, const Tensor &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<Tensor> &attr_value)
Operator &SetAttr(const char_t *name, const OpBytes &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<std::vector<int64_t>> &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<ge::DataType> &attr_value)
Operator &SetAttr(const char_t *name, const ge::DataType &attr_value)
Operator &SetAttr(const char_t *name, const ge::NamedAttrs &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<ge::NamedAttrs> &attr_value)
```

## 参数说明<a name="section24345054"></a>

<a name="table109471639792"></a>
<table><thead align="left"><tr id="row149394014919"><th class="cellrowborder" valign="top" width="20.44%" id="mcps1.1.4.1.1"><p id="p139304018914"><a name="p139304018914"></a><a name="p139304018914"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.68%" id="mcps1.1.4.1.2"><p id="p14932401292"><a name="p14932401292"></a><a name="p14932401292"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="59.88%" id="mcps1.1.4.1.3"><p id="p79414014912"><a name="p79414014912"></a><a name="p79414014912"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9941540490"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p4941240190"><a name="p4941240190"></a><a name="p4941240190"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p10948401299"><a name="p10948401299"></a><a name="p10948401299"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p129413401094"><a name="p129413401094"></a><a name="p129413401094"></a>属性名称。</p>
</td>
</tr>
<tr id="row894104016917"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p4722307573"><a name="p4722307573"></a><a name="p4722307573"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p109484011911"><a name="p109484011911"></a><a name="p109484011911"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p11941740595"><a name="p11941740595"></a><a name="p11941740595"></a>需设置的int64_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row59415401193"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p27401009573"><a name="p27401009573"></a><a name="p27401009573"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p1945400917"><a name="p1945400917"></a><a name="p1945400917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p8944401491"><a name="p8944401491"></a><a name="p8944401491"></a>需设置的int32_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row1294340694"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1975617015715"><a name="p1975617015715"></a><a name="p1975617015715"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p59412404912"><a name="p59412404912"></a><a name="p59412404912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p1994440191"><a name="p1994440191"></a><a name="p1994440191"></a>需设置的uint32_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row394840592"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p16759604571"><a name="p16759604571"></a><a name="p16759604571"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p49416409916"><a name="p49416409916"></a><a name="p49416409916"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p129419401196"><a name="p129419401196"></a><a name="p129419401196"></a>需设置的vector&lt;int64_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row3941040295"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p107621011573"><a name="p107621011573"></a><a name="p107621011573"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p4948401095"><a name="p4948401095"></a><a name="p4948401095"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p10943401910"><a name="p10943401910"></a><a name="p10943401910"></a>需设置的vector&lt;int32_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row19424011911"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1076419095719"><a name="p1076419095719"></a><a name="p1076419095719"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p19413401699"><a name="p19413401699"></a><a name="p19413401699"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p295114019912"><a name="p295114019912"></a><a name="p295114019912"></a>需设置的vector&lt;uint32_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row10951940295"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p12767190195720"><a name="p12767190195720"></a><a name="p12767190195720"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p39519401694"><a name="p39519401694"></a><a name="p39519401694"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p19951040598"><a name="p19951040598"></a><a name="p19951040598"></a>需设置的std::initializer_list&lt;int64_t&gt;&amp;&amp;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row49511401299"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p10770705573"><a name="p10770705573"></a><a name="p10770705573"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p14951401995"><a name="p14951401995"></a><a name="p14951401995"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p129574011916"><a name="p129574011916"></a><a name="p129574011916"></a>需设置的浮点类型的属性值。</p>
</td>
</tr>
<tr id="row1895184011917"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p67736085717"><a name="p67736085717"></a><a name="p67736085717"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p179517401795"><a name="p179517401795"></a><a name="p179517401795"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p79516409914"><a name="p79516409914"></a><a name="p79516409914"></a>需设置的浮点列表类型的属性值。</p>
</td>
</tr>
<tr id="row4959406917"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p377620014578"><a name="p377620014578"></a><a name="p377620014578"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p9958401798"><a name="p9958401798"></a><a name="p9958401798"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p109516401492"><a name="p109516401492"></a><a name="p109516401492"></a>需设置的布尔类型的属性值。</p>
</td>
</tr>
<tr id="row11951401199"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1477910010572"><a name="p1477910010572"></a><a name="p1477910010572"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p19951440399"><a name="p19951440399"></a><a name="p19951440399"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p3951400910"><a name="p3951400910"></a><a name="p3951400910"></a>需设置的布尔列表类型的属性值。</p>
</td>
</tr>
<tr id="row75401427133417"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1578190205710"><a name="p1578190205710"></a><a name="p1578190205710"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p16753829144812"><a name="p16753829144812"></a><a name="p16753829144812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p2541162717344"><a name="p2541162717344"></a><a name="p2541162717344"></a>需设置的AttrValue类型的属性值。</p>
</td>
</tr>
<tr id="row69516401915"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p19784110175717"><a name="p19784110175717"></a><a name="p19784110175717"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p1795840698"><a name="p1795840698"></a><a name="p1795840698"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p3951140894"><a name="p3951140894"></a><a name="p3951140894"></a>需设置的字符串类型的属性值。</p>
</td>
</tr>
<tr id="row4959401999"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p378713095717"><a name="p378713095717"></a><a name="p378713095717"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p169654018914"><a name="p169654018914"></a><a name="p169654018914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p109654014915"><a name="p109654014915"></a><a name="p109654014915"></a>需设置的字符串列表类型的属性值。</p>
</td>
</tr>
<tr id="row796640296"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p57899013575"><a name="p57899013575"></a><a name="p57899013575"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p16967401298"><a name="p16967401298"></a><a name="p16967401298"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p9962401912"><a name="p9962401912"></a><a name="p9962401912"></a>需设置的Tensor类型的属性值。</p>
</td>
</tr>
<tr id="row1093020810149"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p579214085720"><a name="p579214085720"></a><a name="p579214085720"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p13566142313183"><a name="p13566142313183"></a><a name="p13566142313183"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p7636203261811"><a name="p7636203261811"></a><a name="p7636203261811"></a>需设置的Tensor列表类型的属性值。</p>
</td>
</tr>
<tr id="row13215105151414"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p137946085714"><a name="p137946085714"></a><a name="p137946085714"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p1283824181814"><a name="p1283824181814"></a><a name="p1283824181814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p14209134161817"><a name="p14209134161817"></a><a name="p14209134161817"></a>需设置的Bytes，即字节数组类型的属性值，OpBytes即vector&lt;uint8_t&gt;。</p>
</td>
</tr>
<tr id="row1566519597139"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p10665155961318"><a name="p10665155961318"></a><a name="p10665155961318"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p1706192481818"><a name="p1706192481818"></a><a name="p1706192481818"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p84416515289"><a name="p84416515289"></a><a name="p84416515289"></a>需设置的Bytes，即字节数组类型的属性值，指定了字节流的首地址。</p>
</td>
</tr>
<tr id="row1224331110274"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p2244171113278"><a name="p2244171113278"></a><a name="p2244171113278"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p02446117271"><a name="p02446117271"></a><a name="p02446117271"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p024417117271"><a name="p024417117271"></a><a name="p024417117271"></a>需设置的Bytes，即字节数组类型的属性值，指定了字节流的长度。</p>
</td>
</tr>
<tr id="row990216557130"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1813947578"><a name="p1813947578"></a><a name="p1813947578"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p2527152515189"><a name="p2527152515189"></a><a name="p2527152515189"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p10902145571320"><a name="p10902145571320"></a><a name="p10902145571320"></a>需设置的量化数据的属性值。</p>
</td>
</tr>
<tr id="row27691731113418"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p132712465715"><a name="p132712465715"></a><a name="p132712465715"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p728211805117"><a name="p728211805117"></a><a name="p728211805117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p1676911313344"><a name="p1676911313344"></a><a name="p1676911313344"></a>需设置的vector&lt;vector&lt;int64_t&gt;&gt;表示的整型二维列表类型属性值。</p>
</td>
</tr>
<tr id="row136011437183414"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p44217445718"><a name="p44217445718"></a><a name="p44217445718"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p107842081515"><a name="p107842081515"></a><a name="p107842081515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p16601173712348"><a name="p16601173712348"></a><a name="p16601173712348"></a>需设置的vector&lt;ge::DataType&gt;表示的DataType列表类型属性值。</p>
</td>
</tr>
<tr id="row7959163423415"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p24518445714"><a name="p24518445714"></a><a name="p24518445714"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p1845014915110"><a name="p1845014915110"></a><a name="p1845014915110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p15959163416340"><a name="p15959163416340"></a><a name="p15959163416340"></a>需设置的DataType类型的属性值。</p>
</td>
</tr>
<tr id="row61545232407"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1148124165713"><a name="p1148124165713"></a><a name="p1148124165713"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p13252652104015"><a name="p13252652104015"></a><a name="p13252652104015"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p122525525403"><a name="p122525525403"></a><a name="p122525525403"></a>需设置的NamedAttrs类型的属性值。</p>
</td>
</tr>
<tr id="row126521229104014"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p185134165714"><a name="p185134165714"></a><a name="p185134165714"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="19.68%" headers="mcps1.1.4.1.2 "><p id="p103041153174014"><a name="p103041153174014"></a><a name="p103041153174014"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.88%" headers="mcps1.1.4.1.3 "><p id="p130410532403"><a name="p130410532403"></a><a name="p130410532403"></a>需设置的vector&lt;ge::NamedAttrs&gt;表示的NamedAttrs列表类型的属性值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section17778899"></a>

[Operator](Operator.md)对象本身。

## 异常处理<a name="section25792371"></a>

无。

## 约束说明<a name="section30804749"></a>

无。

