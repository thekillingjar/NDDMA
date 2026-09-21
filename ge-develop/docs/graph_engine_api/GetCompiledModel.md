# GetCompiledModel<a name="ZH-CN_TOPIC_0000002508677545"></a>

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

## 功能说明<a name="section15686020"></a>

获取图编译后的序列化模型。

调用本接口前，必须先调用[CompileGraph](CompileGraph-2.md)；用户通过本接口获取内存缓冲区的序列化模型数据ModelBufferData后，后续操作分为两种：

-   用法一：（推荐用法）生成om离线模型
    1.  在线编译：通过GetCompiledModel接口获取编译后的序列化模型。
    2.  调用[aclgrphSaveModel](aclgrphSaveModel.md)接口将序列化模型保存为om离线模型。

        调用GetCompiledModel接口和调用aclgrphSaveModel接口时，用户应确保所使用的CANN软件包版本**一致**，接口内不做校验。

    3.  调用acl**从文件中**加载模型接口aclmdlLoadFromFile，然后使用aclmdlExecute接口执行推理。

-   用法二：不生成om离线模型
    1.  在线编译：通过GetCompiledModel接口获取编译后的序列化模型。
    2.  调用acl**从内存中**加载模型接口aclmdlLoadFromMem，然后使用aclmdlExecute接口执行推理。

推荐使用用法一：因为生成om离线模型后，调用GetCompiledModel接口和调用acl接口的CANN软件包版本号**可以不一致**；而如果使用用法二，由于内存缓冲区的序列化模型数据ModelBufferData不保证版本兼容性，要求调用GetCompiledModel接口和调用acl接口的CANN软件包版本号**一致**，否则可能引发未定义行为。

acl接口详细介绍请参见《应用开发指南 \(C&C++\)》。

## 函数原型<a name="section1610715016501"></a>

```
Status GetCompiledModel(uint32_t graph_id, ModelDataBuffer &model_buffer)
```

## 参数说明<a name="section6956456"></a>

<a name="table0804192414564"></a>
<table><thead align="left"><tr id="row16804112495613"><th class="cellrowborder" valign="top" width="12.33%" id="mcps1.1.4.1.1"><p id="p178045248562"><a name="p178045248562"></a><a name="p178045248562"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.1%" id="mcps1.1.4.1.2"><p id="p4804024195612"><a name="p4804024195612"></a><a name="p4804024195612"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.57000000000001%" id="mcps1.1.4.1.3"><p id="p9804824205619"><a name="p9804824205619"></a><a name="p9804824205619"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row2080482415617"><td class="cellrowborder" valign="top" width="12.33%" headers="mcps1.1.4.1.1 "><p id="p198049244561"><a name="p198049244561"></a><a name="p198049244561"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="12.1%" headers="mcps1.1.4.1.2 "><p id="p1680410248561"><a name="p1680410248561"></a><a name="p1680410248561"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.57000000000001%" headers="mcps1.1.4.1.3 "><p id="p16662184215413"><a name="p16662184215413"></a><a name="p16662184215413"></a>图ID。</p>
</td>
</tr>
<tr id="row18764192714257"><td class="cellrowborder" valign="top" width="12.33%" headers="mcps1.1.4.1.1 "><p id="p11764202792514"><a name="p11764202792514"></a><a name="p11764202792514"></a>model_buffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.1%" headers="mcps1.1.4.1.2 "><p id="p1276482752518"><a name="p1276482752518"></a><a name="p1276482752518"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.57000000000001%" headers="mcps1.1.4.1.3 "><p id="p65451332717"><a name="p65451332717"></a><a name="p65451332717"></a>序列化的模型，保存在内存缓冲区中。详情请参见<a href="ModelBufferData.md">ModelBufferData</a>。</p>
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

<a name="table79991920134013"></a>
<table><thead align="left"><tr id="row69999207406"><th class="cellrowborder" valign="top" width="11.600000000000001%" id="mcps1.1.4.1.1"><p id="p59991020114011"><a name="p59991020114011"></a><a name="p59991020114011"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.3%" id="mcps1.1.4.1.2"><p id="p8999720114017"><a name="p8999720114017"></a><a name="p8999720114017"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.1%" id="mcps1.1.4.1.3"><p id="p20999220184014"><a name="p20999220184014"></a><a name="p20999220184014"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row299982015406"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.1.4.1.1 "><p id="p15999142019406"><a name="p15999142019406"></a><a name="p15999142019406"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.3%" headers="mcps1.1.4.1.2 "><p id="p17999142004019"><a name="p17999142004019"></a><a name="p17999142004019"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="76.1%" headers="mcps1.1.4.1.3 "><p id="p4999162064013"><a name="p4999162064013"></a><a name="p4999162064013"></a>SUCCESS：成功</p>
<p id="p399919209407"><a name="p399919209407"></a><a name="p399919209407"></a>FAILED：失败</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section38092103"></a>

-   变量与转换算子融合会导致图需要重新编译：由于通过GetCompiledModel接口获取序列化模型后无法再次编译，因此若图中包含变量，需将[options参数说明](options参数说明.md)中的ge.exec.variable\_acc参数配置为False，以关闭变量与转换算子的融合功能。
-   关于启用外置权重功能（[options参数说明](options参数说明.md)中的ge.externalWeight）后的权重文件管理，需注意以下要点：
    -   默认行为：当开启外置权重功能时，系统会默认将权重文件写入一个临时目录（具体路径可参考ge.externalWeight说明）。需要注意的是，在GeSession对象析构时，该临时目录及其下的文件将被自动清理删除。
    -   操作建议：为避免权重文件在GeSession析构后丢失，用户可以参考以下任一方法进行持久化保存：

        -   指定专用目录：建议在初始化时，通过配置ge.externalWeightDir参数，直接指定一个专用的、非临时的路径用于存放权重文件。
        -   提前备份文件：在GeSession对象析构之前，通过应用程序主动将临时路径下的权重文件拷贝至其他指定安全目录。
        -   保存离线模型：在GeSession析构之前，调用aclgrphSaveModel接口，将已编译好的模型（包含权重）序列化保存为独立的om离线模型文件。

        通过以上方式，可以有效管理外置权重文件的生命周期，确保其不会因GeSession的销毁而意外丢失。

