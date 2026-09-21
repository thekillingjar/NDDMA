# RunGraphAsync<a name="ZH-CN_TOPIC_0000002508677541"></a>

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

异步运行指定ID对应的Graph图，输出运行结果。

-   本接口与[RunGraph](RunGraph-10.md)/[RunGraphWithStreamAsync](RunGraphWithStreamAsync-12.md)互斥，若在调用本接口前未执行[LoadGraph](LoadGraph-6.md)完成图加载，则本接口将自动调用LoadGraph以完成加载。
-   本接口与RunGraph均为执行指定id对应的图，并输出结果，区别于RunGraph的是，该接口：
    -   异步运行。
    -   用户通过回调函数RunAsyncCallbackV2获取图计算结果，即用户自行定义函数RunAsyncCallbackV2。该回调函数，当Status为SUCCESS时，即可处理数据。

## 函数原型<a name="section1831611148519"></a>

```
Status RunGraphAsync(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, RunAsyncCallbackV2 callback)
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="9.4%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.7%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="81.89999999999999%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="9.4%" headers="mcps1.1.4.1.1 "><p id="p6011995"><a name="p6011995"></a><a name="p6011995"></a>graph_id</p>
</td>
<td class="cellrowborder" valign="top" width="8.7%" headers="mcps1.1.4.1.2 "><p id="p17209562"><a name="p17209562"></a><a name="p17209562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="81.89999999999999%" headers="mcps1.1.4.1.3 "><p id="p270712592519"><a name="p270712592519"></a><a name="p270712592519"></a>子图对应的ID。</p>
</td>
</tr>
<tr id="row0533152292412"><td class="cellrowborder" valign="top" width="9.4%" headers="mcps1.1.4.1.1 "><p id="p153312242411"><a name="p153312242411"></a><a name="p153312242411"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="8.7%" headers="mcps1.1.4.1.2 "><p id="p05331222182416"><a name="p05331222182416"></a><a name="p05331222182416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="81.89999999999999%" headers="mcps1.1.4.1.3 "><p id="p6626238163513"><a name="p6626238163513"></a><a name="p6626238163513"></a>计算图输入数据，可以位于Host，也可以位于Device上。</p>
<p id="p6155459165710"><a name="p6155459165710"></a><a name="p6155459165710"></a>如果数据位于Host上，而执行在device上，用户输入进入数据队列时，需要对每个输入做一次内存的申请和拷贝，输入较大或者较多时可能存在性能瓶颈。</p>
</td>
</tr>
<tr id="row7751173518241"><td class="cellrowborder" valign="top" width="9.4%" headers="mcps1.1.4.1.1 "><p id="p275193515244"><a name="p275193515244"></a><a name="p275193515244"></a>callback</p>
</td>
<td class="cellrowborder" valign="top" width="8.7%" headers="mcps1.1.4.1.2 "><p id="p1475110353246"><a name="p1475110353246"></a><a name="p1475110353246"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="81.89999999999999%" headers="mcps1.1.4.1.3 "><p id="p10751123519246"><a name="p10751123519246"></a><a name="p10751123519246"></a>当前子图对应的回调函数，输出是GE申请的Host内存。</p>
<a name="screen666619444496"></a><a name="screen666619444496"></a><pre class="screen" codetype="Cpp" id="screen666619444496">using Status = uint32_t;
using RunAsyncCallbackV2 = std::function&lt;void(Status, std::vector&lt;gert::Tensor&gt; &amp;)&gt;;</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="9.28%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.61%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="82.11%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="9.28%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.61%" headers="mcps1.1.4.1.2 "><p id="p31747823"><a name="p31747823"></a><a name="p31747823"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="82.11%" headers="mcps1.1.4.1.3 "><p id="p141243178210"><a name="p141243178210"></a><a name="p141243178210"></a>GE_CLI_GE_NOT_INITIALIZED：GE未初始化。</p>
<p id="p918814449444"><a name="p918814449444"></a><a name="p918814449444"></a>SUCCESS：异步运行图成功。</p>
<p id="p91883446443"><a name="p91883446443"></a><a name="p91883446443"></a>FAILED：异步运行图失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

## 调用示例<a name="section68291386"></a>

1.  调用接口AddGraph加载子图。

    ```
    std::map <AscendString, AscendString> options;
    ge::GeSession *session = new GeSession(options);
    
    uint32_t graph_id = 0;
    ge::Graph graph;
    Status ret = session->AddGraph(graph_id, graph);
    ```

2.  用户自定义RunAsyncCallbackV2，来决定如何处理数据，例如：

    ```
    void CallBack(Status result, std::vector<gert::Tensor> &out_tensor) {
    if(result == ge::SUCCESS) {
    // 读取out_tensor数据， 用户根据需求处理数据；
    for(auto &tensor : out_tensor) {
      auto data = tensor.GetAddr();
      int64_t length = tensor.GetSize();
      }
    }
    }
    ```

3.  定义好指定图的输入数据const std::vector<gert::Tensor\> &inputs。
4.  调用接口RunGraphAsync\(graph\_id, inputs, CallBack\)。

