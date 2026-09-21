# GetRegisteredIrDef<a name="ZH-CN_TOPIC_0000002520800877"></a>

## 产品支持情况<a name="section1993214622115"></a>

<a name="zh-cn_topic_0000001265245842_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265245842_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001265245842_p1883113061818"><a name="zh-cn_topic_0000001265245842_p1883113061818"></a><a name="zh-cn_topic_0000001265245842_p1883113061818"></a><span id="zh-cn_topic_0000001265245842_ph20833205312295"><a name="zh-cn_topic_0000001265245842_ph20833205312295"></a><a name="zh-cn_topic_0000001265245842_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001265245842_p783113012187"><a name="zh-cn_topic_0000001265245842_p783113012187"></a><a name="zh-cn_topic_0000001265245842_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265245842_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265245842_p48327011813"><a name="zh-cn_topic_0000001265245842_p48327011813"></a><a name="zh-cn_topic_0000001265245842_p48327011813"></a><span id="zh-cn_topic_0000001265245842_ph583230201815"><a name="zh-cn_topic_0000001265245842_ph583230201815"></a><a name="zh-cn_topic_0000001265245842_ph583230201815"></a><term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265245842_p108715341013"><a name="zh-cn_topic_0000001265245842_p108715341013"></a><a name="zh-cn_topic_0000001265245842_p108715341013"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265245842_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265245842_p14832120181815"><a name="zh-cn_topic_0000001265245842_p14832120181815"></a><a name="zh-cn_topic_0000001265245842_p14832120181815"></a><span id="zh-cn_topic_0000001265245842_ph1483216010188"><a name="zh-cn_topic_0000001265245842_ph1483216010188"></a><a name="zh-cn_topic_0000001265245842_ph1483216010188"></a><term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265245842_p19948143911820"><a name="zh-cn_topic_0000001265245842_p19948143911820"></a><a name="zh-cn_topic_0000001265245842_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <ge/ge\_api.h\>
-   库文件：libge\_runner.so

## 功能说明<a name="section44282627"></a>

获取注册的IR（Intermediate Representation）算子原型定义信息。

## 函数原型<a name="section1831611148519"></a>

```
ge::Status GetRegisteredIrDef(const char *op_type, std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs, std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs, std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="11.33%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.52%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="80.15%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row7616101515116"><td class="cellrowborder" valign="top" width="11.33%" headers="mcps1.1.4.1.1 "><p id="p199962615327"><a name="p199962615327"></a><a name="p199962615327"></a>op_type</p>
</td>
<td class="cellrowborder" valign="top" width="8.52%" headers="mcps1.1.4.1.2 "><p id="p16019272324"><a name="p16019272324"></a><a name="p16019272324"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.15%" headers="mcps1.1.4.1.3 "><p id="p192754106131"><a name="p192754106131"></a><a name="p192754106131"></a><span>算子类型</span>。</p>
</td>
</tr>
<tr id="row17472816"><td class="cellrowborder" valign="top" width="11.33%" headers="mcps1.1.4.1.1 "><p id="p6011995"><a name="p6011995"></a><a name="p6011995"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="8.52%" headers="mcps1.1.4.1.2 "><p id="p17209562"><a name="p17209562"></a><a name="p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.15%" headers="mcps1.1.4.1.3 "><p id="p1441810513219"><a name="p1441810513219"></a><a name="p1441810513219"></a><span>算子的</span>输入名称及类型，类型包括required/dynamic/optional。</p>
</td>
</tr>
<tr id="row17126854103114"><td class="cellrowborder" valign="top" width="11.33%" headers="mcps1.1.4.1.1 "><p id="p81271554123112"><a name="p81271554123112"></a><a name="p81271554123112"></a>outputs</p>
</td>
<td class="cellrowborder" valign="top" width="8.52%" headers="mcps1.1.4.1.2 "><p id="p5127105419314"><a name="p5127105419314"></a><a name="p5127105419314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.15%" headers="mcps1.1.4.1.3 "><p id="p270712592519"><a name="p270712592519"></a><a name="p270712592519"></a><span>算子的</span>输出名称及类型，类型包括required/dynamic。</p>
</td>
</tr>
<tr id="row0533152292412"><td class="cellrowborder" valign="top" width="11.33%" headers="mcps1.1.4.1.1 "><p id="p153312242411"><a name="p153312242411"></a><a name="p153312242411"></a>attrs</p>
</td>
<td class="cellrowborder" valign="top" width="8.52%" headers="mcps1.1.4.1.2 "><p id="p05331222182416"><a name="p05331222182416"></a><a name="p05331222182416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.15%" headers="mcps1.1.4.1.3 "><p id="p1088102316362"><a name="p1088102316362"></a><a name="p1088102316362"></a>算子属性名称及属性类型，类型包括（与已有接口保持风格一致，数据类型前带有VT_前缀）INT/FLOAT/STRING/BOOL/DATA_TYPE/TENSOR/NAMED_ATTRS/LIST_INT/LIST_FLOAT/LIST_STRING/LIST_BOOL/LIST_TENSOR/BYTES/LIST_LIST_INT/LIST_NAMED_ATTRS。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="13.239999999999998%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.280000000000001%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="74.48%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="13.239999999999998%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.280000000000001%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="74.48%" headers="mcps1.1.4.1.3 "><p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>SUCCESS：查询成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：查询失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

