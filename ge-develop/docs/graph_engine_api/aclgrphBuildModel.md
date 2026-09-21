# aclgrphBuildModel<a name="ZH-CN_TOPIC_0000001312725921"></a>

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

## 功能说明<a name="section15686020"></a>

将输入的Graph编译为适配昇腾AI处理器的离线模型，并保存到内存缓冲区。

## 函数原型<a name="section1610715016501"></a>

> [!NOTE]说明
>数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus aclgrphBuildModel(const ge::Graph &graph, const std::map<std::string, std::string> &build_options, ModelBufferData &model)
graphStatus aclgrphBuildModel(const ge::Graph &graph, const std::map<AscendString, AscendString> &build_options, ModelBufferData &model)
```

## 参数说明<a name="section6956456"></a>

<a name="table33761356"></a>
<table><thead align="left"><tr id="row27598891"><th class="cellrowborder" valign="top" width="12.15%" id="mcps1.1.4.1.1"><p id="p20917673"><a name="p20917673"></a><a name="p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.42%" id="mcps1.1.4.1.2"><p id="p16609919"><a name="p16609919"></a><a name="p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.43%" id="mcps1.1.4.1.3"><p id="p59995477"><a name="p59995477"></a><a name="p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row27795493"><td class="cellrowborder" valign="top" width="12.15%" headers="mcps1.1.4.1.1 "><p id="p36842478"><a name="p36842478"></a><a name="p36842478"></a>graph</p>
</td>
<td class="cellrowborder" valign="top" width="8.42%" headers="mcps1.1.4.1.2 "><p id="p31450711"><a name="p31450711"></a><a name="p31450711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.43%" headers="mcps1.1.4.1.3 "><p id="p183101154193614"><a name="p183101154193614"></a><a name="p183101154193614"></a>待编译的Graph。</p>
</td>
</tr>
<tr id="row1988716421732"><td class="cellrowborder" valign="top" width="12.15%" headers="mcps1.1.4.1.1 "><p id="p188871421336"><a name="p188871421336"></a><a name="p188871421336"></a>build_options</p>
</td>
<td class="cellrowborder" valign="top" width="8.42%" headers="mcps1.1.4.1.2 "><p id="p8887124212317"><a name="p8887124212317"></a><a name="p8887124212317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.43%" headers="mcps1.1.4.1.3 "><p id="p15590142217205"><a name="p15590142217205"></a><a name="p15590142217205"></a>graph级别配置参数。</p>
<p id="p1523641218382"><a name="p1523641218382"></a><a name="p1523641218382"></a>配置参数map映射表，key为参数类型，value为参数值，均为字符串格式，用于描述离线模型编译配置信息。</p>
<p id="p172061244411"><a name="p172061244411"></a><a name="p172061244411"></a>map中的配置参数请参见<a href="aclgrphBuildModel支持的配置参数.md">aclgrphBuildModel支持的配置参数</a>。</p>
</td>
</tr>
<tr id="row3121045136"><td class="cellrowborder" valign="top" width="12.15%" headers="mcps1.1.4.1.1 "><p id="p1012194518314"><a name="p1012194518314"></a><a name="p1012194518314"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.42%" headers="mcps1.1.4.1.2 "><p id="p13126451836"><a name="p13126451836"></a><a name="p13126451836"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="79.43%" headers="mcps1.1.4.1.3 "><p id="p17256091945"><a name="p17256091945"></a><a name="p17256091945"></a>编译生成的离线模型缓存，模型保存在内存缓冲区中。详情请参见<a href="ModelBufferData.md">ModelBufferData</a>。</p>
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

## 返回值说明<a name="section62608109"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="9.959999999999999%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.16%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="80.88%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="9.959999999999999%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>graphStatus<strong id="b3686432181711"><a name="b3686432181711"></a><a name="b3686432181711"></a></strong></p>
</td>
<td class="cellrowborder" valign="top" width="80.88%" headers="mcps1.1.4.1.3 "><p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section38092103"></a>

-   **对于aclgrphBuildModel和aclgrphBuildInitialize中重复的编译配置参数，建议不要重复配置，如果设置重复，则以aclgrphBuildModel传入的为准**。
-   使用aclgrphBuildModel接口传入build\_options参数时，多张图场景下，如果传入的参数为ge::ir\_option::PRECISION\_MODE或者ge::ir\_option::PRECISION\_MODE\_V2，多张图设置的参数值需要相同。
-   使用aclgrphBuildModel接口编译的离线模型，保存在内存缓冲区中：

    -   如果希望将内存缓冲区中的模型保存为离线模型文件xx.om，则需要调用[aclgrphSaveModel](aclgrphSaveModel.md)接口，序列化保存离线模型到文件中。后续使用acl接口进行推理业务，需要使用**从文件中**加载模型的接口，例如aclmdlLoadFromFile，然后使用aclmdlExecute接口执行推理。
    -   如果离线模型保存在内存缓冲区：

        后续使用acl接口进行推理业务时，需要使用**从内存中**加载模型的接口，例如aclmdlLoadFromMem，然后使用aclmdlExecute接口执行推理。

    acl接口详细介绍请参见《应用开发指南 \(C&C++\)》。

