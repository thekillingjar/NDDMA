# LoadGraph<a name="ZH-CN_TOPIC_0000002508557563"></a>

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

-   头文件：\#include <ge/ge\_api\_v2.h\>
-   库文件：libge\_runner\_v2.so

## 功能说明<a name="section44282627"></a>

加载图并为其执行做准备，包括申请与管理图运行所需的内存及计算流等资源。

说明：若在调用本接口前未执行[CompileGraph](CompileGraph-2.md)完成图编译，则本接口将自动调用CompileGraph以完成编译。

## 函数原型<a name="section1831611148519"></a>

```
Status LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options, void *stream) const
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="9.55%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.25%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="82.19999999999999%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="9.55%" headers="mcps1.1.4.1.1 "><p id="p6011995"><a name="p6011995"></a><a name="p6011995"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.1.4.1.2 "><p id="p17209562"><a name="p17209562"></a><a name="p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.19999999999999%" headers="mcps1.1.4.1.3 "><p id="p34829302"><a name="p34829302"></a><a name="p34829302"></a>要执行Graph对应的ID。</p>
</td>
</tr>
<tr id="row0115745088"><td class="cellrowborder" valign="top" width="9.55%" headers="mcps1.1.4.1.1 "><p id="p611624515811"><a name="p611624515811"></a><a name="p611624515811"></a><span>options</span></p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.1.4.1.2 "><p id="p1211615453816"><a name="p1211615453816"></a><a name="p1211615453816"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.19999999999999%" headers="mcps1.1.4.1.3 "><p id="p365171011499"><a name="p365171011499"></a><a name="p365171011499"></a><span>执行阶段可能用到的</span><span>option</span>s。map表，key为参数类型，value为参数值，描述Graph配置信息。</p>
<p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p15921183410"></a>一般情况下可不填，与<a href="GEInitializeV2.md">GEInitializeV2</a>传入的全局options保持一致。</p>
<p id="p1082917817426"><a name="p1082917817426"></a><a name="p1082917817426"></a><span>key和value类型为AscendString</span>，如需单独配置当前Graph的配置信息时，可以通过此参数配置，支持的配置项请参见<a href="options参数说明.md">options参数说明</a>&gt;<strong id="b8463133714459"><a name="b8463133714459"></a><a name="b8463133714459"></a>ge.exec.frozenInputIndexes和ge.exec.hostInputIndexes</strong>，当前只支持配置上述两个参数。</p>
</td>
</tr>
<tr id="row188211176016"><td class="cellrowborder" valign="top" width="9.55%" headers="mcps1.1.4.1.1 "><p id="p48218171605"><a name="p48218171605"></a><a name="p48218171605"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.1.4.1.2 "><p id="p18821717703"><a name="p18821717703"></a><a name="p18821717703"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="82.19999999999999%" headers="mcps1.1.4.1.3 "><p id="p13523121420545"><a name="p13523121420545"></a><a name="p13523121420545"></a><span id="ph22835178118"><a name="ph22835178118"></a><a name="ph22835178118"></a>acl</span>接口<span id="ph32324712536"><a name="ph32324712536"></a><a name="ph32324712536"></a>“aclrtCreateStream”</span>创建的流，也可以设置为nullptr。当传入有效值时，若在加载过程中需要向流中下发任务，会下发到指定流上。</p>
<a name="ul4414132614298"></a><a name="ul4414132614298"></a><ul id="ul4414132614298"><li>若与<a href="RunGraphWithStreamAsync-12.md">RunGraphWithStreamAsync</a>接口配合使用，建议传入有效值。此时，通过LoadGraph加载的Stream与RunGraphWithStreamAsync运行时使用的Stream推荐为同一条流。若非同一条流，则需在LoadGraph后，对加载使用的Stream调用<span id="ph1043416291381"><a name="ph1043416291381"></a><a name="ph1043416291381"></a>acl</span>流同步接口<span id="ph16761715104518"><a name="ph16761715104518"></a><a name="ph16761715104518"></a>“aclrtSynchronizeStream”</span>完成同步。</li><li>若与<a href="RunGraph-10.md">RunGraph</a>或<a href="RunGraphAsync-11.md">RunGraphAsync</a>接口配合使用，建议传入nullptr以简化流程。若传入有效值，则需在LoadGraph后、RunGraph/RunGraphAsync前调用<span id="ph1948519653918"><a name="ph1948519653918"></a><a name="ph1948519653918"></a>“aclrtSynchronizeStream”</span>完成流同步，以确保加载任务完成。</li></ul>
<p id="p6623172414496"><a name="p6623172414496"></a><a name="p6623172414496"></a><span id="ph979216568013"><a name="ph979216568013"></a><a name="ph979216568013"></a>acl</span>接口详细说明请参见<span id="ph262362413495"><a name="ph262362413495"></a><a name="ph262362413495"></a>《应用开发指南 (C&amp;C++)》</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="9.84%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.18%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="81.98%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.18%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="81.98%" headers="mcps1.1.4.1.3 "><p id="p141243178210"><a name="p141243178210"></a><a name="p141243178210"></a>GE_CLI_SESS_RUN_FAILED：执行子图时序列化转换失败。</p>
<p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>SUCCESS：执行子图成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：执行子图失败。</p>
</td>
</tr>
</tbody>
</table>

