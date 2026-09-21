# CreateInput<a name="ZH-CN_TOPIC_0000002488265168"></a>

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

## 功能说明<a name="section44282627"></a>

创建输入节点。

## 函数原型<a name="section1831611148519"></a>

-   创建图输入节点

    ```
    EsTensorHolder CreateInput(int64_t index,const char *name,const char *type)
    ```

-   创建默认输入节点，从0开始计数，节点命名为input\_\{index\}

    ```
    EsTensorHolder CreateInput(int64_t index)
    ```

-   创建指定名称的输入节点

    ```
    EsTensorHolder CreateInput(int64_t index, const char *name)
    ```

-   创建具有完整信息的输入节点

    ```
    EsTensorHolder CreateInput(int64_t index, const char *name, ge::DataType data_type, ge::Format format, const std::vector<int64_t> &shape)
    ```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="10.73%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.47%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p1583161612618"><a name="p1583161612618"></a><a name="p1583161612618"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p165822162612"><a name="p165822162612"></a><a name="p165822162612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p185820162619"><a name="p185820162619"></a><a name="p185820162619"></a>输入节点的索引，从0开始计数。</p>
</td>
</tr>
<tr id="row8834747194419"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p15835147154411"><a name="p15835147154411"></a><a name="p15835147154411"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p3835164784420"><a name="p3835164784420"></a><a name="p3835164784420"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p2835104714416"><a name="p2835104714416"></a><a name="p2835104714416"></a>输入节点的名称，如果为nullptr则使用默认名称input_{index}。</p>
</td>
</tr>
<tr id="row867904916441"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p367974954413"><a name="p367974954413"></a><a name="p367974954413"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p16679124964417"><a name="p16679124964417"></a><a name="p16679124964417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p1767984919444"><a name="p1767984919444"></a><a name="p1767984919444"></a>输入节点的类型字符串，如果为nullptr则默认为Data。</p>
</td>
</tr>
<tr id="row65974418531"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p419985724913"><a name="p419985724913"></a><a name="p419985724913"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p5199857104918"><a name="p5199857104918"></a><a name="p5199857104918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p319965712499"><a name="p319965712499"></a><a name="p319965712499"></a>张量格式。</p>
</td>
</tr>
<tr id="row177052439535"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p1811775919496"><a name="p1811775919496"></a><a name="p1811775919496"></a>data_type</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p1117115920494"><a name="p1117115920494"></a><a name="p1117115920494"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p161171959104918"><a name="p161171959104918"></a><a name="p161171959104918"></a>数据类型。</p>
</td>
</tr>
<tr id="row044216454533"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p188521206501"><a name="p188521206501"></a><a name="p188521206501"></a>shape</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p785290205018"><a name="p785290205018"></a><a name="p785290205018"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p485210011507"><a name="p485210011507"></a><a name="p485210011507"></a>张量形状向量，如果为空则创建标量。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.06%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.55999999999999%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.1.4.1.2 "><p id="p1157618141266"><a name="p1157618141266"></a><a name="p1157618141266"></a>EsTensorHolder</p>
</td>
<td class="cellrowborder" valign="top" width="76.55999999999999%" headers="mcps1.1.4.1.3 "><p id="p95757141768"><a name="p95757141768"></a><a name="p95757141768"></a>返回创建的输入张量持有者，失败时返回无效的EsTensorHolder。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

## 调用示例<a name="section16305113853313"></a>

-   创建图输入节点

    ```
    EsGraphBuilder builder("graph_name");
    auto t1 = builder.CreateInput(0, "input0", "Data");
    ```

-   创建默认输入节点，从0开始计数，节点命名为input\_\{index\}

    ```
    EsGraphBuilder builder("graph_name");
    auto tensor = builder.CreateInput(1);
    ```

-   创建指定名称的输入节点

    ```
    EsGraphBuilder builder("graph_name");
    auto tensor = builder.CreateInput(2, "input2");
    ```

-   创建具有完整信息的输入节点

    ```
    EsGraphBuilder builder("graph_name");
    auto tensor = builder.CreateInput(3, "input3", ge::DT_INT32, ge::FORMAT_NCHW, {2, 2});
    ```

