# GetOutputAttr<a name="ZH-CN_TOPIC_0000002499520448"></a>

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

## 功能说明<a name="section23101557"></a>

根据属性名称获取算子输出Tensor对应的属性值。

## 函数原型<a name="section45525325619"></a>

```
graphStatus GetOutputAttr(const int32_t index, const char_t *name, AscendString &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, AscendString &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, int64_t &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, int64_t &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, int32_t &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, int32_t &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, uint32_t &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, uint32_t &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, bool &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, bool &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, float32_t &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, float32_t &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<AscendString> &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<AscendString> &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<int64_t> &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<int64_t> &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<int32_t> &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<int32_t> &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<uint32_t> &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<uint32_t> &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<bool> &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<bool> &attr_value) const
graphStatus GetOutputAttr(const int32_t index, const char_t *name, std::vector<float32_t> &attr_value) const
graphStatus GetOutputAttr(const char_t *dst_name, const char_t *name, std::vector<float32_t> &attr_value) const
```

## 参数说明<a name="section6587426"></a>

<a name="table109471639792"></a>
<table><thead align="left"><tr id="row149394014919"><th class="cellrowborder" valign="top" width="20.44%" id="mcps1.1.4.1.1"><p id="p139304018914"><a name="p139304018914"></a><a name="p139304018914"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="24.68%" id="mcps1.1.4.1.2"><p id="p14932401292"><a name="p14932401292"></a><a name="p14932401292"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="54.879999999999995%" id="mcps1.1.4.1.3"><p id="p79414014912"><a name="p79414014912"></a><a name="p79414014912"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row9941540490"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p4941240190"><a name="p4941240190"></a><a name="p4941240190"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p10948401299"><a name="p10948401299"></a><a name="p10948401299"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p129413401094"><a name="p129413401094"></a><a name="p129413401094"></a>属性名称。</p>
</td>
</tr>
<tr id="row18562114263216"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p83171645132712"><a name="p83171645132712"></a><a name="p83171645132712"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p1231774512717"><a name="p1231774512717"></a><a name="p1231774512717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p831764511275"><a name="p831764511275"></a><a name="p831764511275"></a>输出索引。</p>
</td>
</tr>
<tr id="row1901644173214"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p31521041192816"><a name="p31521041192816"></a><a name="p31521041192816"></a>dst_name</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p18152154192819"><a name="p18152154192819"></a><a name="p18152154192819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p1815274182812"><a name="p1815274182812"></a><a name="p1815274182812"></a>输出边名称。</p>
</td>
</tr>
<tr id="row894104016917"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p4722307573"><a name="p4722307573"></a><a name="p4722307573"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p109484011911"><a name="p109484011911"></a><a name="p109484011911"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p11941740595"><a name="p11941740595"></a><a name="p11941740595"></a>获取到的int64_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row59415401193"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p27401009573"><a name="p27401009573"></a><a name="p27401009573"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p846392518339"><a name="p846392518339"></a><a name="p846392518339"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p8944401491"><a name="p8944401491"></a><a name="p8944401491"></a>获取到的int32_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row1294340694"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1975617015715"><a name="p1975617015715"></a><a name="p1975617015715"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p3480192518336"><a name="p3480192518336"></a><a name="p3480192518336"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p1994440191"><a name="p1994440191"></a><a name="p1994440191"></a>获取到的uint32_t表示的整型类型属性值。</p>
</td>
</tr>
<tr id="row394840592"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p16759604571"><a name="p16759604571"></a><a name="p16759604571"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p94985259330"><a name="p94985259330"></a><a name="p94985259330"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p129419401196"><a name="p129419401196"></a><a name="p129419401196"></a>获取到的vector&lt;int64_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row3941040295"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p107621011573"><a name="p107621011573"></a><a name="p107621011573"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p1550014258332"><a name="p1550014258332"></a><a name="p1550014258332"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p10943401910"><a name="p10943401910"></a><a name="p10943401910"></a>获取到的vector&lt;int32_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row19424011911"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1076419095719"><a name="p1076419095719"></a><a name="p1076419095719"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p13502325183310"><a name="p13502325183310"></a><a name="p13502325183310"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p295114019912"><a name="p295114019912"></a><a name="p295114019912"></a>获取到的vector&lt;uint32_t&gt;表示的整型列表类型属性值。</p>
</td>
</tr>
<tr id="row49511401299"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p10770705573"><a name="p10770705573"></a><a name="p10770705573"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p165041025123312"><a name="p165041025123312"></a><a name="p165041025123312"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p129574011916"><a name="p129574011916"></a><a name="p129574011916"></a>获取到的浮点类型的属性值。</p>
</td>
</tr>
<tr id="row1895184011917"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p67736085717"><a name="p67736085717"></a><a name="p67736085717"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p15506225163311"><a name="p15506225163311"></a><a name="p15506225163311"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p79516409914"><a name="p79516409914"></a><a name="p79516409914"></a>获取到的浮点列表类型的属性值。</p>
</td>
</tr>
<tr id="row4959406917"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p377620014578"><a name="p377620014578"></a><a name="p377620014578"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p1510172518332"><a name="p1510172518332"></a><a name="p1510172518332"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p109516401492"><a name="p109516401492"></a><a name="p109516401492"></a>获取到的布尔类型的属性值。</p>
</td>
</tr>
<tr id="row11951401199"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p1477910010572"><a name="p1477910010572"></a><a name="p1477910010572"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p155134256337"><a name="p155134256337"></a><a name="p155134256337"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p3951400910"><a name="p3951400910"></a><a name="p3951400910"></a>获取到的布尔列表类型的属性值。</p>
</td>
</tr>
<tr id="row69516401915"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p19784110175717"><a name="p19784110175717"></a><a name="p19784110175717"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p1951522514334"><a name="p1951522514334"></a><a name="p1951522514334"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p3951140894"><a name="p3951140894"></a><a name="p3951140894"></a>获取到的字符串类型的属性值。</p>
</td>
</tr>
<tr id="row4959401999"><td class="cellrowborder" valign="top" width="20.44%" headers="mcps1.1.4.1.1 "><p id="p378713095717"><a name="p378713095717"></a><a name="p378713095717"></a>attr_value</p>
</td>
<td class="cellrowborder" valign="top" width="24.68%" headers="mcps1.1.4.1.2 "><p id="p205175255338"><a name="p205175255338"></a><a name="p205175255338"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="54.879999999999995%" headers="mcps1.1.4.1.3 "><p id="p109654014915"><a name="p109654014915"></a><a name="p109654014915"></a>获取到的字符串列表类型的属性值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section59286834"></a>

graphStatus类型：找到对应属性，返回GRAPH\_SUCCESS，否则返回GRAPH\_FAILED。

## 异常处理<a name="section63819464"></a>

无。

## 约束说明<a name="section37504268"></a>

无。

