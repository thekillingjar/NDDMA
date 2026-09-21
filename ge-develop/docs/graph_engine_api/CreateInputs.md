# CreateInputs<a name="ZH-CN_TOPIC_0000002520345011"></a>

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

创建指定数量的输入节点。

## 函数原型<a name="section1831611148519"></a>

-   创建指定数量的输入节点，返回数组，适用于编译时的结构化绑定

    ```
    template <size_t inputs_num>
    std::array<EsTensorHolder, inputs_num> CreateInputs(size_t start_index = 0)
    ```

-   创建指定数量的输入节点，返回容器，适用于运行时动态绑定

    ```
    std::vector<EsTensorHolder> CreateInputs(size_t inputs_num, size_t start_index = 0)
    ```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="13.919999999999998%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.540000000000001%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.53999999999999%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="13.919999999999998%" headers="mcps1.1.4.1.1 "><p id="p1583161612618"><a name="p1583161612618"></a><a name="p1583161612618"></a>inputs_num</p>
</td>
<td class="cellrowborder" valign="top" width="11.540000000000001%" headers="mcps1.1.4.1.2 "><p id="p165822162612"><a name="p165822162612"></a><a name="p165822162612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.1.4.1.3 "><p id="p185820162619"><a name="p185820162619"></a><a name="p185820162619"></a>输入节点数量。</p>
</td>
</tr>
<tr id="row598351934620"><td class="cellrowborder" valign="top" width="13.919999999999998%" headers="mcps1.1.4.1.1 "><p id="p11983201954613"><a name="p11983201954613"></a><a name="p11983201954613"></a>start_index</p>
</td>
<td class="cellrowborder" valign="top" width="11.540000000000001%" headers="mcps1.1.4.1.2 "><p id="p7983181910468"><a name="p7983181910468"></a><a name="p7983181910468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.1.4.1.3 "><p id="p99831319164615"><a name="p99831319164615"></a><a name="p99831319164615"></a>输入节点起始索引，如果不为0表示前面已经创建了其他输入节点，整体输入节点的索引应该从0开始连续递增。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.06%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="27.810000000000002%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="61.129999999999995%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="27.810000000000002%" headers="mcps1.1.4.1.2 "><p id="p1157618141266"><a name="p1157618141266"></a><a name="p1157618141266"></a>std::array&lt;EsTensorHolder, inputs_num&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="61.129999999999995%" headers="mcps1.1.4.1.3 "><p id="p1551612344710"><a name="p1551612344710"></a><a name="p1551612344710"></a>返回创建的输入张量持有者数组。</p>
</td>
</tr>
<tr id="row48493550102"><td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.1 "><p id="p5468135911016"><a name="p5468135911016"></a><a name="p5468135911016"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="27.810000000000002%" headers="mcps1.1.4.1.2 "><p id="p17335473479"><a name="p17335473479"></a><a name="p17335473479"></a>std::vector&lt;EsTensorHolder&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="61.129999999999995%" headers="mcps1.1.4.1.3 "><p id="p51955312479"><a name="p51955312479"></a><a name="p51955312479"></a>返回创建的输入张量持有者容器。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

## 调用示例<a name="section16305113853313"></a>

-   创建指定数量的输入节点，返回数组，适用于编译时的结构化绑定

    ```
    // 1. 创建图构建器（EsGraphBuilder）
    EsGraphBuilder builder("graph_name");
    // 2. 添加 2 个输入节点，使用c++17的结构化绑定到data0和data1
    EsTensorHolder [data0, data1] = builder.CreateInputs<2>();
    ```

-   创建指定数量的输入节点，返回容器，适用于运行时动态绑定

    ```
    // 1. 创建图构建器（EsGraphBuilder）
    EsGraphBuilder builder("graph_name");
    // 2. 添加 2 个输入节点
    std::vector<EsTensorHolder> inputs = builder.CreateInputs(2);
    ```

