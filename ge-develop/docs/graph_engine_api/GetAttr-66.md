# GetAttr<a name="ZH-CN_TOPIC_0000002499520438"></a>

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

## 功能说明<a name="section21529561"></a>

根据属性名称获取对应的属性值。

## 函数原型<a name="section721703420111"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus GetAttr(const std::string &name, int64_t &attr_value) const
graphStatus GetAttr(const std::string &name, int32_t &attr_value) const
graphStatus GetAttr(const std::string &name, uint32_t &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<int64_t> &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<int32_t> &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<uint32_t> &attr_value) const
graphStatus GetAttr(const std::string &name, float32_t &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<float32_t> &attr_value) const
graphStatus GetAttr(const std::string &name, AttrValue &attr_value) const
graphStatus GetAttr(const std::string &name, std::string &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<std::string> &attr_value) const
graphStatus GetAttr(const std::string &name, bool &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<bool> &attr_value) const
graphStatus GetAttr(const std::string &name, Tensor &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<Tensor> &attr_value) const
graphStatus GetAttr(const std::string &name, OpBytes &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<std::vector<int64_t>> &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<ge::DataType> &attr_value) const
graphStatus GetAttr(const std::string &name, ge::DataType &attr_value) const
graphStatus GetAttr(const std::string &name, ge::NamedAttrs &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<ge::NamedAttrs> &attr_value) const
graphStatus GetAttr(const char_t *name, int64_t &attr_value) const
graphStatus GetAttr(const char_t *name, int32_t &attr_value) const
graphStatus GetAttr(const char_t *name, uint32_t &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<int64_t> &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<int32_t> &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<uint32_t> &attr_value) const
graphStatus GetAttr(const char_t *name, float32_t &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<float32_t> &attr_value) const
graphStatus GetAttr(const char_t *name, AttrValue &attr_value) const
graphStatus GetAttr(const char_t *name, AscendString &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<AscendString> &attr_values) const
graphStatus GetAttr(const char_t *name, bool &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<bool> &attr_value) const
graphStatus GetAttr(const char_t *name, Tensor &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<Tensor> &attr_value) const
graphStatus GetAttr(const char_t *name, OpBytes &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<std::vector<int64_t>> &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<ge::DataType> &attr_value) const
graphStatus GetAttr(const char_t *name, ge::DataType &attr_value) const
graphStatus GetAttr(const char_t *name, ge::NamedAttrs &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<ge::NamedAttrs> &attr_value) const
```

## 参数说明<a name="section59548322"></a>

<a name="table109471639792"></a>
<table><thead align="left"><tr id="row149394014919"><th class="cellrowborder" valign="top" width="20.810000000000002%" id="mcps1.1.4.1.1"><p id="p139304018914"><a name="p139304018914"></a><a name="p139304018914"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.96%" id="mcps1.1.4.1.2"><p id="p14932401292"><a name="p14932401292"></a><a name="p14932401292"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="53.23%" id="mcps1.1.4.1.3"><p id="p79414014912"><a name="p79414014912"></a><a name="p79414014912"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9941540490"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p4941240190"><a name="p4941240190"></a><a name="p4941240190"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p10948401299"><a name="p10948401299"></a><a name="p10948401299"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p129413401094"><a name="p129413401094"></a><a name="p129413401094"></a>属性名称。</p>
</td>
</tr>
<tr id="row894104016917"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p1645462765810"><a name="p1645462765810"></a><a name="p1645462765810"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p109484011911"><a name="p109484011911"></a><a name="p109484011911"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p11941740595"><a name="p11941740595"></a><a name="p11941740595"></a>返回的int64_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row59415401193"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p15469527175818"><a name="p15469527175818"></a><a name="p15469527175818"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p1945400917"><a name="p1945400917"></a><a name="p1945400917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p8944401491"><a name="p8944401491"></a><a name="p8944401491"></a>返回的int32_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row1294340694"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p548352710581"><a name="p548352710581"></a><a name="p548352710581"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p59412404912"><a name="p59412404912"></a><a name="p59412404912"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p1994440191"><a name="p1994440191"></a><a name="p1994440191"></a>返回的uint32_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row394840592"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p348692710585"><a name="p348692710585"></a><a name="p348692710585"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p49416409916"><a name="p49416409916"></a><a name="p49416409916"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p129419401196"><a name="p129419401196"></a><a name="p129419401196"></a>返回的vector&lt;int64_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row3941040295"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p9488132710586"><a name="p9488132710586"></a><a name="p9488132710586"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p4948401095"><a name="p4948401095"></a><a name="p4948401095"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p10943401910"><a name="p10943401910"></a><a name="p10943401910"></a>返回的vector&lt;int32_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row19424011911"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p649032745818"><a name="p649032745818"></a><a name="p649032745818"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p19413401699"><a name="p19413401699"></a><a name="p19413401699"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p295114019912"><a name="p295114019912"></a><a name="p295114019912"></a>返回的vector&lt;uint32_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row49511401299"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p749202711586"><a name="p749202711586"></a><a name="p749202711586"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p14951401995"><a name="p14951401995"></a><a name="p14951401995"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p129574011916"><a name="p129574011916"></a><a name="p129574011916"></a>返回的浮点类型的属性值。</p>
</td>
</tr>
<tr id="row1895184011917"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p9494427115811"><a name="p9494427115811"></a><a name="p9494427115811"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p179517401795"><a name="p179517401795"></a><a name="p179517401795"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p79516409914"><a name="p79516409914"></a><a name="p79516409914"></a>返回的浮点列表类型的属性值。</p>
</td>
</tr>
<tr id="row1485137175618"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p12496202735811"><a name="p12496202735811"></a><a name="p12496202735811"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p999985125617"><a name="p999985125617"></a><a name="p999985125617"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p2541162717344"><a name="p2541162717344"></a><a name="p2541162717344"></a>返回的AttrValue类型的属性值。</p>
</td>
</tr>
<tr id="row4959406917"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p6498112713586"><a name="p6498112713586"></a><a name="p6498112713586"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p9958401798"><a name="p9958401798"></a><a name="p9958401798"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p109516401492"><a name="p109516401492"></a><a name="p109516401492"></a>返回的布尔类型的属性值。</p>
</td>
</tr>
<tr id="row11951401199"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p75005278584"><a name="p75005278584"></a><a name="p75005278584"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p19951440399"><a name="p19951440399"></a><a name="p19951440399"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p3951400910"><a name="p3951400910"></a><a name="p3951400910"></a>返回的布尔列表类型的属性值。</p>
</td>
</tr>
<tr id="row69516401915"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p12502527105814"><a name="p12502527105814"></a><a name="p12502527105814"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p1795840698"><a name="p1795840698"></a><a name="p1795840698"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p3951140894"><a name="p3951140894"></a><a name="p3951140894"></a>返回的字符串类型的属性值。</p>
</td>
</tr>
<tr id="row4959401999"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p1650412715817"><a name="p1650412715817"></a><a name="p1650412715817"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p169654018914"><a name="p169654018914"></a><a name="p169654018914"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p109654014915"><a name="p109654014915"></a><a name="p109654014915"></a>返回的字符串列表类型的属性值。</p>
</td>
</tr>
<tr id="row796640296"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p1450620273583"><a name="p1450620273583"></a><a name="p1450620273583"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p16967401298"><a name="p16967401298"></a><a name="p16967401298"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p9962401912"><a name="p9962401912"></a><a name="p9962401912"></a>返回的Tensor类型的属性值。</p>
</td>
</tr>
<tr id="row1093020810149"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p1150810279585"><a name="p1150810279585"></a><a name="p1150810279585"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p13566142313183"><a name="p13566142313183"></a><a name="p13566142313183"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p7636203261811"><a name="p7636203261811"></a><a name="p7636203261811"></a>返回的Tensor列表类型的属性值。</p>
</td>
</tr>
<tr id="row13215105151414"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p125107277584"><a name="p125107277584"></a><a name="p125107277584"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p1283824181814"><a name="p1283824181814"></a><a name="p1283824181814"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p14209134161817"><a name="p14209134161817"></a><a name="p14209134161817"></a>返回的Bytes，即字节数组类型的属性值，OpBytes即vector&lt;uint8_t&gt;。</p>
</td>
</tr>
<tr id="row990216557130"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p155121427115820"><a name="p155121427115820"></a><a name="p155121427115820"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p2527152515189"><a name="p2527152515189"></a><a name="p2527152515189"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p10902145571320"><a name="p10902145571320"></a><a name="p10902145571320"></a>返回的量化数据的属性值。</p>
</td>
</tr>
<tr id="row168841040115610"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p35141527175814"><a name="p35141527175814"></a><a name="p35141527175814"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p19471105416562"><a name="p19471105416562"></a><a name="p19471105416562"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p1676911313344"><a name="p1676911313344"></a><a name="p1676911313344"></a>返回的vector&lt;vector&lt;int64_t&gt;&gt;表示的整型二维列表类型属性值。</p>
</td>
</tr>
<tr id="row072724655618"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p1151611276584"><a name="p1151611276584"></a><a name="p1151611276584"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p108611654155611"><a name="p108611654155611"></a><a name="p108611654155611"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p16601173712348"><a name="p16601173712348"></a><a name="p16601173712348"></a>返回的vector&lt;ge::DataType&gt;表示的DataType列表类型属性值。</p>
</td>
</tr>
<tr id="row126061443195614"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p85181827125817"><a name="p85181827125817"></a><a name="p85181827125817"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p1325575575610"><a name="p1325575575610"></a><a name="p1325575575610"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p15959163416340"><a name="p15959163416340"></a><a name="p15959163416340"></a>返回的DataType类型的属性值。</p>
</td>
</tr>
<tr id="row10307192010475"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p5520727115810"><a name="p5520727115810"></a><a name="p5520727115810"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p18766128124715"><a name="p18766128124715"></a><a name="p18766128124715"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p276652894716"><a name="p276652894716"></a><a name="p276652894716"></a>返回的vector&lt;ge::NamedAttrs&gt;表示的NamedAttrs列表类型属性值。</p>
</td>
</tr>
<tr id="row457992314476"><td class="cellrowborder" valign="top" width="20.810000000000002%" headers="mcps1.1.4.1.1 "><p id="p05221527125819"><a name="p05221527125819"></a><a name="p05221527125819"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="25.96%" headers="mcps1.1.4.1.2 "><p id="p0767428174719"><a name="p0767428174719"></a><a name="p0767428174719"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="53.23%" headers="mcps1.1.4.1.3 "><p id="p6767142812475"><a name="p6767142812475"></a><a name="p6767142812475"></a>返回的NamedAttrs类型的属性值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section66172851"></a>

graphStatus类型：找到对应name，返回GRAPH\_SUCCESS，否则返回GRAPH\_FAILED。

## 异常处理<a name="section58684749"></a>

无。

## 约束说明<a name="section58400697"></a>

无。

