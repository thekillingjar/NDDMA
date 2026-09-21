# GetAttr<a name="ZH-CN_TOPIC_0000002484436788"></a>

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

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <graph/gnode.h\>
-   库文件：libgraph.so

## 功能说明<a name="section10161546"></a>

获取指定属性名字的值。

## 函数原型<a name="section182148481597"></a>

```
graphStatus GetAttr(const AscendString &name, int64_t &attr_value) const
graphStatus GetAttr(const AscendString &name, int32_t &attr_value) const
graphStatus GetAttr(const AscendString &name, uint32_t &attr_value) const
graphStatus GetAttr(const AscendString &name, float32_t &attr_value) const
graphStatus GetAttr(const AscendString &name, AscendString &attr_value) const
graphStatus GetAttr(const AscendString &name, bool &attr_value) const
graphStatus GetAttr(const AscendString &name, Tensor &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<int64_t> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<int32_t> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<uint32_t> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<float32_t> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const
graphStatus GetAttr(const AscendString &name, std::vector<bool> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<Tensor> &attr_value) const
graphStatus GetAttr(const AscendString &name, OpBytes &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<std::vector<int64_t>> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<ge::DataType> &attr_value) const
graphStatus GetAttr(const AscendString &name, ge::DataType &attr_value) const
graphStatus GetAttr(const AscendString &name, AttrValue &attr_value) const
```

## 参数说明<a name="section24345054"></a>

<a name="table109471639792"></a>
<table><thead align="left"><tr id="row149394014919"><th class="cellrowborder" valign="top" width="13.87%" id="mcps1.1.4.1.1"><p id="p139304018914"><a name="p139304018914"></a><a name="p139304018914"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.68%" id="mcps1.1.4.1.2"><p id="p14932401292"><a name="p14932401292"></a><a name="p14932401292"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.44999999999999%" id="mcps1.1.4.1.3"><p id="p79414014912"><a name="p79414014912"></a><a name="p79414014912"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row9941540490"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p4941240190"><a name="p4941240190"></a><a name="p4941240190"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p10948401299"><a name="p10948401299"></a><a name="p10948401299"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p129413401094"><a name="p129413401094"></a><a name="p129413401094"></a>属性名称。</p>
</td>
</tr>
<tr id="row894104016917"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p11632133410610"><a name="p11632133410610"></a><a name="p11632133410610"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p109484011911"><a name="p109484011911"></a><a name="p109484011911"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p11941740595"><a name="p11941740595"></a><a name="p11941740595"></a>返回的int64_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row59415401193"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p12647143412615"><a name="p12647143412615"></a><a name="p12647143412615"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p13415103010410"><a name="p13415103010410"></a><a name="p13415103010410"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p8944401491"><a name="p8944401491"></a><a name="p8944401491"></a>返回的int32_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row1294340694"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p36599348612"><a name="p36599348612"></a><a name="p36599348612"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p1542053014113"><a name="p1542053014113"></a><a name="p1542053014113"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p1994440191"><a name="p1994440191"></a><a name="p1994440191"></a>返回的uint32_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row15830191145610"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p1666118341761"><a name="p1666118341761"></a><a name="p1666118341761"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p204221430204117"><a name="p204221430204117"></a><a name="p204221430204117"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p1214121835619"><a name="p1214121835619"></a><a name="p1214121835619"></a>返回的浮点类型float的属性值。</p>
</td>
</tr>
<tr id="row177413812590"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p7663203415618"><a name="p7663203415618"></a><a name="p7663203415618"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p184241730144117"><a name="p184241730144117"></a><a name="p184241730144117"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p3951140894"><a name="p3951140894"></a><a name="p3951140894"></a>返回的字符串类型AscendString的属性值。</p>
</td>
</tr>
<tr id="row165036303563"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p136654342615"><a name="p136654342615"></a><a name="p136654342615"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p17426153015411"><a name="p17426153015411"></a><a name="p17426153015411"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p109516401492"><a name="p109516401492"></a><a name="p109516401492"></a>返回的布尔类型bool的属性值。</p>
</td>
</tr>
<tr id="row1370378105718"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p46669341966"><a name="p46669341966"></a><a name="p46669341966"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p54287307417"><a name="p54287307417"></a><a name="p54287307417"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p9962401912"><a name="p9962401912"></a><a name="p9962401912"></a>返回的Tensor类型的属性值。</p>
</td>
</tr>
<tr id="row394840592"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p10668153411613"><a name="p10668153411613"></a><a name="p10668153411613"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p843003015414"><a name="p843003015414"></a><a name="p843003015414"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p129419401196"><a name="p129419401196"></a><a name="p129419401196"></a>返回的vector&lt;int64_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row3941040295"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p1267015345612"><a name="p1267015345612"></a><a name="p1267015345612"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p1143393012411"><a name="p1143393012411"></a><a name="p1143393012411"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p10943401910"><a name="p10943401910"></a><a name="p10943401910"></a>返回的vector&lt;int32_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row19424011911"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p7671183417617"><a name="p7671183417617"></a><a name="p7671183417617"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p04351130124113"><a name="p04351130124113"></a><a name="p04351130124113"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p295114019912"><a name="p295114019912"></a><a name="p295114019912"></a>返回的vector&lt;uint32_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row1432515222581"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p136735341469"><a name="p136735341469"></a><a name="p136735341469"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p7438153013417"><a name="p7438153013417"></a><a name="p7438153013417"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p79516409914"><a name="p79516409914"></a><a name="p79516409914"></a>返回的浮点列表类型vector&lt;float&gt;的属性值。</p>
</td>
</tr>
<tr id="row151351754175813"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p1867513341563"><a name="p1867513341563"></a><a name="p1867513341563"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p114403304410"><a name="p114403304410"></a><a name="p114403304410"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p109654014915"><a name="p109654014915"></a><a name="p109654014915"></a>返回的字符串列表类型vector&lt;ge::AscendString&gt;的属性值。</p>
</td>
</tr>
<tr id="row468917596583"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p9677934467"><a name="p9677934467"></a><a name="p9677934467"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p044213309418"><a name="p044213309418"></a><a name="p044213309418"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p3951400910"><a name="p3951400910"></a><a name="p3951400910"></a>返回的布尔列表类型vector&lt;bool&gt;的属性值。</p>
</td>
</tr>
<tr id="row1253595616584"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p1968016349616"><a name="p1968016349616"></a><a name="p1968016349616"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p13444630194114"><a name="p13444630194114"></a><a name="p13444630194114"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p7636203261811"><a name="p7636203261811"></a><a name="p7636203261811"></a>返回的Tensor列表类型vector&lt;Tensor&gt;的属性值。</p>
</td>
</tr>
<tr id="row1373555115587"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p96819343619"><a name="p96819343619"></a><a name="p96819343619"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p19446930104110"><a name="p19446930104110"></a><a name="p19446930104110"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p14209134161817"><a name="p14209134161817"></a><a name="p14209134161817"></a>返回的Bytes，即字节数组类型的属性值，OpBytes即vector&lt;uint8_t&gt;。</p>
</td>
</tr>
<tr id="row3343749205820"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p18683103418616"><a name="p18683103418616"></a><a name="p18683103418616"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p144823044119"><a name="p144823044119"></a><a name="p144823044119"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p1676911313344"><a name="p1676911313344"></a><a name="p1676911313344"></a>返回的vector&lt;vector&lt;int64_t&gt;&gt;表示的整型二维列表类型属性值。</p>
</td>
</tr>
<tr id="row1323194710212"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p268411348610"><a name="p268411348610"></a><a name="p268411348610"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p19450230104110"><a name="p19450230104110"></a><a name="p19450230104110"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p16601173712348"><a name="p16601173712348"></a><a name="p16601173712348"></a>返回的vector&lt;ge::DataType&gt;表示的DataType列表类型属性值。</p>
</td>
</tr>
<tr id="row177595414216"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p206865347610"><a name="p206865347610"></a><a name="p206865347610"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p8452113018418"><a name="p8452113018418"></a><a name="p8452113018418"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p15959163416340"><a name="p15959163416340"></a><a name="p15959163416340"></a>返回的DataType类型的属性值。</p>
</td>
</tr>
<tr id="row127511452128"><td class="cellrowborder" valign="top" width="13.87%" headers="mcps1.1.4.1.1 "><p id="p136881134269"><a name="p136881134269"></a><a name="p136881134269"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="10.68%" headers="mcps1.1.4.1.2 "><p id="p7454630174115"><a name="p7454630174115"></a><a name="p7454630174115"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.44999999999999%" headers="mcps1.1.4.1.3 "><p id="p2541162717344"><a name="p2541162717344"></a><a name="p2541162717344"></a>返回的AttrValue类型的属性值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section17778899"></a>

<a name="table23900756"></a>
<table><thead align="left"><tr id="row12896286"><th class="cellrowborder" valign="top" width="13.750000000000002%" id="mcps1.1.4.1.1"><p id="p37966223"><a name="p37966223"></a><a name="p37966223"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.989999999999998%" id="mcps1.1.4.1.2"><p id="p55365254"><a name="p55365254"></a><a name="p55365254"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="73.26%" id="mcps1.1.4.1.3"><p id="p55400552"><a name="p55400552"></a><a name="p55400552"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row58259743"><td class="cellrowborder" valign="top" width="13.750000000000002%" headers="mcps1.1.4.1.1 "><p id="p21418771"><a name="p21418771"></a><a name="p21418771"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.989999999999998%" headers="mcps1.1.4.1.2 "><p id="p73026092813"><a name="p73026092813"></a><a name="p73026092813"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="73.26%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0235751031_p918814449444"><a name="zh-cn_topic_0235751031_p918814449444"></a><a name="zh-cn_topic_0235751031_p918814449444"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="zh-cn_topic_0235751031_p91883446443"><a name="zh-cn_topic_0235751031_p91883446443"></a><a name="zh-cn_topic_0235751031_p91883446443"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section30804749"></a>

无

