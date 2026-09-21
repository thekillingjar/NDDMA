# aclgrphBundleSaveModel<a name="ZH-CN_TOPIC_0000001969984924"></a>

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

将离线模型序列化并保存到指定文件中。

-   与[aclgrphSaveModel](aclgrphSaveModel.md)接口的区别是，该接口适用于权重更新场景，且通过该接口生成的模型，如果要使用acl进行推理，需要使用aclmdlBundleLoadFromFile或aclmdlBundleLoadFromMem接口加载模型，接口详细介绍请参见《应用开发指南 \(Python\)》。
-   通过[aclgrphBundleBuildModel](aclgrphBundleBuildModel.md)接口生成离线模型缓存后才可以使用aclgrphBundleSaveModel接口落盘。

## 函数原型<a name="section1831611148519"></a>

```
graphStatus aclgrphBundleSaveModel(const char_t *output_file, const ModelBufferData &model)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="10.97%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.93%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.10000000000001%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="10.97%" headers="mcps1.1.4.1.1 "><p id="p36842478"><a name="p36842478"></a><a name="p36842478"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="9.93%" headers="mcps1.1.4.1.2 "><p id="p31450711"><a name="p31450711"></a><a name="p31450711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.4.1.3 "><p id="p97921059122013"><a name="p97921059122013"></a><a name="p97921059122013"></a>aclgrphBundleBuildModel生成的离线模型缓冲数据。详情请参见<a href="ModelBufferData.md">ModelBufferData</a>。</p>
<a name="screen17756917183"></a><a name="screen17756917183"></a><pre class="screen" codetype="Cpp" id="screen17756917183">struct ModelBufferData
{
  std::shared_ptr&lt;uint8_t&gt; data = nullptr;
  uint32_t length;
};</pre>
<p id="p7467641112117"><a name="p7467641112117"></a><a name="p7467641112117"></a>其中data指向生成的模型数据，length代表该模型的实际大小。</p>
</td>
</tr>
<tr id="row18648104064012"><td class="cellrowborder" valign="top" width="10.97%" headers="mcps1.1.4.1.1 "><p id="p188871421336"><a name="p188871421336"></a><a name="p188871421336"></a>output_file</p>
</td>
<td class="cellrowborder" valign="top" width="9.93%" headers="mcps1.1.4.1.2 "><p id="p8887124212317"><a name="p8887124212317"></a><a name="p8887124212317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.4.1.3 "><p id="p7669105718244"><a name="p7669105718244"></a><a name="p7669105718244"></a>保存离线模型的文件名。生成的离线模型文件名会自动以.om后缀结尾，例如<em id="i18771103364915"><a name="i18771103364915"></a><a name="i18771103364915"></a>ir_build_sample.om</em>或者</p>
<p id="p5380383333"><a name="p5380383333"></a><a name="p5380383333"></a><em id="i1668313557015"><a name="i1668313557015"></a><a name="i1668313557015"></a>ir_build_sample_linux_x86_64</em>.om，若om文件名中包含操作系统及架构，则该om文件只能在该操作系统及架构的运行环境中使用。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.78%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.19%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="77.03%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.78%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.19%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="77.03%" headers="mcps1.1.4.1.3 "><p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

若生成的om模型文件名中含操作系统及架构，但操作系统及其架构与模型运行环境不一致时，需要与OPTION\_HOST\_ENV\_OS、OPTION\_HOST\_ENV\_CPU参数配合使用，设置模型运行环境的操作系统类型及架构。参数具体介绍请参见[aclgrphBuildInitialize支持的配置参数](aclgrphBuildInitialize支持的配置参数.md)。

## 调用示例<a name="section117112710367"></a>

```
// 包含的头文件：
#include "ge_ir_build.h" 
#include "ge_api_types.h"

// 1.生成Graph
Graph origin_graph("Irorigin_graph");
GenGraph(origin_graph);

// 2.创建完Graph以后，通过aclgrphBuildInitialize接口进行系统初始化，并申请资源
std::map<AscendString, AscendString> global_options;
auto status = aclgrphBuildInitialize(global_options);

// 3. 生成权重可更新的图，包括三部分：权重初始化图，权重更新图，推理图
// 权重初始化是可选步骤，根据业务场景由用户判断是否需要包含权重初始化图，不包含的情况下，可节省模型加载所需的Device内存
WeightRefreshableGraphs weight_refreshable_graphs;
std::vector<AscendString> const_names;
const_names.emplace_back(AscendString("const_1"));
const_names.emplace_back(AscendString("const_2"));
status = aclgrphConvertToWeightRefreshableGraphs(origin_graph, const_names, weight_refreshable_graphs);
if (status != GRAPH_SUCCESS) {
    cout << "aclgrphConvertToWeightRefreshableGraphs failed!" << endl;
    aclgrphBuildFinalize();
    return -1;
}

// 4.将权重可更新的图，编译成离线模型，并保存在内存缓冲区中
std::map<AscendString, AscendString> options;
std::vector<ge::GraphWithOptions> graph_and_options;
// 推理图
graph_and_options.push_back(GraphWithOptions{weight_refreshable_graphs.infer_graph, options});
// 权重初始化图
graph_and_options.push_back(GraphWithOptions{weight_refreshable_graphs.var_init_graph, options});
// 权重更新图
graph_and_options.push_back(GraphWithOptions{weight_refreshable_graphs.var_update_graph, options});
ge::ModelBufferData model;
status = aclgrphBundleBuildModel(graph_and_options, model);
if (status != GRAPH_SUCCESS) {
    cout << "aclgrphBundleBuildModel failed" << endl;
    aclgrphBuildFinalize();
    return -1;
}

// 5.将内存缓冲区中的模型保存为离线模型文件
const char *file="./xxx" ;
status = aclgrphBundleSaveModel(file, model);
if (status != GRAPH_SUCCESS) {
    cout << "aclgrphBundleSaveModel failed" << endl;
    aclgrphBuildFinalize();
    return -1;
}
// 6.构图进程结束，释放资源
aclgrphBuildFinalize();
```

