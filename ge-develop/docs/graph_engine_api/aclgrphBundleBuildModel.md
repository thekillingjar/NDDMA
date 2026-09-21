# aclgrphBundleBuildModel<a name="ZH-CN_TOPIC_0000002006425485"></a>

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

-   头文件：\#include <ge/ge\_ir\_build.h\>
-   库文件：libge\_compiler.so

## 功能说明<a name="section44282627"></a>

将输入的一组Graph编译为适配昇腾AI处理器的离线模型。

与[aclgrphBuildModel](aclgrphBuildModel.md)接口的区别是，该接口适用于权重更新场景。通过aclgrphBundleBuildModel接口生成离线模型缓存后，需要使用[aclgrphBundleSaveModel](aclgrphBundleSaveModel.md)接口落盘。

## 函数原型<a name="section1831611148519"></a>

```
graphStatus aclgrphBundleBuildModel(const std::vector<ge::GraphWithOptions> &graph_with_options, ModelBufferData &model)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="14.89%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.520000000000001%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.59%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="14.89%" headers="mcps1.1.4.1.1 "><p id="p10119654151218"><a name="p10119654151218"></a><a name="p10119654151218"></a>graph_with_options</p>
</td>
<td class="cellrowborder" valign="top" width="9.520000000000001%" headers="mcps1.1.4.1.2 "><p id="p4119165411127"><a name="p4119165411127"></a><a name="p4119165411127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.1.4.1.3 "><p id="p2011812542123"><a name="p2011812542123"></a><a name="p2011812542123"></a>待编译的一组图和编译参数。该入参为一个结构体，包括如下参数：</p>
<a name="screen1795016913466"></a><a name="screen1795016913466"></a><pre class="screen" codetype="Cpp" id="screen1795016913466">struct GraphWithOptions {
  ge::Graph graph;
  std::map&lt;AscendString, AscendString&gt; build_options;
};</pre>
<p id="p1236213432119"><a name="p1236213432119"></a><a name="p1236213432119"></a>一组图包括：权重初始化图，权重更新图，推理图，其中只有推理图支持设置如下<strong id="b18874133710124"><a name="b18874133710124"></a><a name="b18874133710124"></a>options</strong>参数。</p>
<p id="p19890541113"><a name="p19890541113"></a><a name="p19890541113"></a>可以通过传入<strong id="b20651183311212"><a name="b20651183311212"></a><a name="b20651183311212"></a>options</strong>参数配置离线模型编译配置信息，当前支持的配置参数请参见<a href="aclgrphBuildModel支持的配置参数.md">aclgrphBuildModel支持的配置参数</a>。配置举例：</p>
<a name="screen47236304110"></a><a name="screen47236304110"></a><pre class="screen" codetype="Cpp" id="screen47236304110">void PrepareOptions(std::map&lt;AscendString, AscendString&gt;&amp; options) {
    options.insert({
        {ge::ir_option::EXEC_DISABLE_REUSED_MEMORY, "1"} // close reuse memory
        });
}</pre>
</td>
</tr>
<tr id="row195948302317"><td class="cellrowborder" valign="top" width="14.89%" headers="mcps1.1.4.1.1 "><p id="p1012194518314"><a name="p1012194518314"></a><a name="p1012194518314"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="9.520000000000001%" headers="mcps1.1.4.1.2 "><p id="p13126451836"><a name="p13126451836"></a><a name="p13126451836"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.1.4.1.3 "><p id="p17256091945"><a name="p17256091945"></a><a name="p17256091945"></a>编译生成的离线模型缓存。详情请参见<a href="ModelBufferData.md">ModelBufferData</a>。</p>
<a name="screen17756917183"></a><a name="screen17756917183"></a><pre class="screen" codetype="Cpp" id="screen17756917183">struct ModelBufferData
{
  std::shared_ptr&lt;uint8_t&gt; data = nullptr;
  uint64_t length;
};</pre>
<p id="p7467641112117"><a name="p7467641112117"></a><a name="p7467641112117"></a>其中data指向生成的模型数据，length代表该模型的实际大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.78%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.089999999999998%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="75.13%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.78%" headers="mcps1.1.4.1.1 "><p id="p12754132812343"><a name="p12754132812343"></a><a name="p12754132812343"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.089999999999998%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>graphStatus<strong id="b3686432181711"><a name="b3686432181711"></a><a name="b3686432181711"></a></strong></p>
</td>
<td class="cellrowborder" valign="top" width="75.13%" headers="mcps1.1.4.1.3 "><p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

该接口内多个图如果涉及共享同名variable算子，接口内部会进行variable算子加速融合，建议通过[aclgrphConvertToWeightRefreshableGraphs](aclgrphConvertToWeightRefreshableGraphs.md)接口生成，否则可能会出现variable算子格式不一致的问题。

## 调用示例<a name="section1042752318432"></a>

完整调用示例请参见[调用示例](aclgrphBundleSaveModel.md#section117112710367)。

