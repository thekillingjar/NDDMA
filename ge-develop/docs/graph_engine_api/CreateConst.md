# CreateConst<a name="ZH-CN_TOPIC_0000002488105202"></a>

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

创建Const算子。

## 函数原型<a name="section1831611148519"></a>

-   创建int64类型的Const算子

    ```
    EsTensorHolder CreateConst(const std::vector<int64_t> &value, const std::vector<int64_t> dims)
    ```

-   创建int32类型的Const算子

    ```
    EsTensorHolder CreateConst(const std::vector<int32_t> &value, const std::vector<int64_t> dims)
    ```

-   创建uint64类型的Const算子

    ```
    EsTensorHolder CreateConst(const std::vector<uint64_t> &value, const std::vector<int64_t> dims)
    ```

-   创建uint32类型的Const算子

    ```
    EsTensorHolder CreateConst(const std::vector<uint32_t> &value, const std::vector<int64_t> dims)
    ```

-   创建float类型的Const算子

    ```
    EsTensorHolder CreateConst(const std::vector<float> &value, const std::vector<int64_t> dims)
    ```

-   创建指定类型、格式和维度的Const算子（通用模板方法）

    ```
    template <typename T>
    EsTensorHolder CreateConst(const std::vector<T> &value, const std::vector<int64_t> &dims, ge::DataType dt,ge::Format format = FORMAT_ND)
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
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p1583161612618"><a name="p1583161612618"></a><a name="p1583161612618"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p165822162612"><a name="p165822162612"></a><a name="p165822162612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p185820162619"><a name="p185820162619"></a><a name="p185820162619"></a>张量数据向量。</p>
</td>
</tr>
<tr id="row17536111912527"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p3536151995212"><a name="p3536151995212"></a><a name="p3536151995212"></a>dims</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p653771965220"><a name="p653771965220"></a><a name="p653771965220"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p1853720195524"><a name="p1853720195524"></a><a name="p1853720195524"></a>张量维度向量。</p>
</td>
</tr>
<tr id="row12237112151712"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p575871961719"><a name="p575871961719"></a><a name="p575871961719"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p2075841911177"><a name="p2075841911177"></a><a name="p2075841911177"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p4758121912176"><a name="p4758121912176"></a><a name="p4758121912176"></a>张量数据类型。</p>
</td>
</tr>
<tr id="row488351316170"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p1161032611576"><a name="p1161032611576"></a><a name="p1161032611576"></a>dt</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p15610826115713"><a name="p15610826115713"></a><a name="p15610826115713"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p16610122645712"><a name="p16610122645712"></a><a name="p16610122645712"></a>张量的数据类型。</p>
</td>
</tr>
<tr id="row252317155175"><td class="cellrowborder" valign="top" width="10.73%" headers="mcps1.1.4.1.1 "><p id="p12449112814575"><a name="p12449112814575"></a><a name="p12449112814575"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.1.4.1.2 "><p id="p16449202819573"><a name="p16449202819573"></a><a name="p16449202819573"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.47%" headers="mcps1.1.4.1.3 "><p id="p9449192845716"><a name="p9449192845716"></a><a name="p9449192845716"></a>张量格式，默认为FORMAT_ND。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.06%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.74%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="75.2%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.74%" headers="mcps1.1.4.1.2 "><p id="p1157618141266"><a name="p1157618141266"></a><a name="p1157618141266"></a>EsTensorHolder</p>
</td>
<td class="cellrowborder" valign="top" width="75.2%" headers="mcps1.1.4.1.3 "><p id="p95757141768"><a name="p95757141768"></a><a name="p95757141768"></a>返回创建的Const的张量持有者算子，失败时返回无效的EsTensorHolder。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

## 调用示例<a name="section16305113853313"></a>

-   创建int64类型的Const算子：

    ```
    EsGraphBuilder builder("test_graph");
    std::vector<int64_t> dims = {3};
    std::vector<int64_t> vec64 = {1, 2, 3};
    auto c1 = builder.CreateConst(vec64, dims);
    ```

-   创建指定类型、格式和维度的Const算子：

    ```
    EsGraphBuilder builder("test_graph");
    std::vector<float> vecf = {1.1, 2.0, 3.2, 4.4};
    std::vector<int64_t> dims = {4};
    auto c1 = builder.CreateConst<float>(vecf, dims, ge::DT_FLOAT);
    ```

