# EsCreateConst<a name="ZH-CN_TOPIC_0000002519215171"></a>

## 功能说明<a name="section44282627"></a>

创建指定类型的Const算子（原始指针版本）。

## 函数原型<a name="section1831611148519"></a>

```
template <typename T>
EsCTensorHolder *EsCreateConst(EsCGraphBuilder *graph, const T *value, const int64_t *dims, int64_t dim_num, ge::DataType dt, ge::Format format = FORMAT_ND) 
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="11.790000000000001%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.6%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.61%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="11.790000000000001%" headers="mcps1.1.4.1.1 "><p id="p1583161612618"><a name="p1583161612618"></a><a name="p1583161612618"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.2 "><p id="p165822162612"><a name="p165822162612"></a><a name="p165822162612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.61%" headers="mcps1.1.4.1.3 "><p id="p185820162619"><a name="p185820162619"></a><a name="p185820162619"></a>张量数据类型。</p>
</td>
</tr>
<tr id="row107217586567"><td class="cellrowborder" valign="top" width="11.790000000000001%" headers="mcps1.1.4.1.1 "><p id="p35821516364"><a name="p35821516364"></a><a name="p35821516364"></a>graph</p>
</td>
<td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.2 "><p id="p1885152903812"><a name="p1885152903812"></a><a name="p1885152903812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.61%" headers="mcps1.1.4.1.3 "><p id="p11581191613619"><a name="p11581191613619"></a><a name="p11581191613619"></a>图构建器指针。</p>
</td>
</tr>
<tr id="row821633613614"><td class="cellrowborder" valign="top" width="11.790000000000001%" headers="mcps1.1.4.1.1 "><p id="p15811716166"><a name="p15811716166"></a><a name="p15811716166"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.2 "><p id="p037113012383"><a name="p037113012383"></a><a name="p037113012383"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.61%" headers="mcps1.1.4.1.3 "><p id="p25811516365"><a name="p25811516365"></a><a name="p25811516365"></a>张量数据指针。</p>
</td>
</tr>
<tr id="row978412503374"><td class="cellrowborder" valign="top" width="11.790000000000001%" headers="mcps1.1.4.1.1 "><p id="p13785650143712"><a name="p13785650143712"></a><a name="p13785650143712"></a>dims</p>
</td>
<td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.2 "><p id="p93917313385"><a name="p93917313385"></a><a name="p93917313385"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.61%" headers="mcps1.1.4.1.3 "><p id="p778575013716"><a name="p778575013716"></a><a name="p778575013716"></a>张量维度数组指针。</p>
</td>
</tr>
<tr id="row1841115063812"><td class="cellrowborder" valign="top" width="11.790000000000001%" headers="mcps1.1.4.1.1 "><p id="p14411160123811"><a name="p14411160123811"></a><a name="p14411160123811"></a>dim_num</p>
</td>
<td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.2 "><p id="p2597193111387"><a name="p2597193111387"></a><a name="p2597193111387"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.61%" headers="mcps1.1.4.1.3 "><p id="p94121204389"><a name="p94121204389"></a><a name="p94121204389"></a>维度数量。</p>
</td>
</tr>
<tr id="row1609131313386"><td class="cellrowborder" valign="top" width="11.790000000000001%" headers="mcps1.1.4.1.1 "><p id="p4609131311381"><a name="p4609131311381"></a><a name="p4609131311381"></a>dt</p>
</td>
<td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.2 "><p id="p61860328382"><a name="p61860328382"></a><a name="p61860328382"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.61%" headers="mcps1.1.4.1.3 "><p id="p260961317381"><a name="p260961317381"></a><a name="p260961317381"></a>张量的数据类型。</p>
</td>
</tr>
<tr id="row187975194383"><td class="cellrowborder" valign="top" width="11.790000000000001%" headers="mcps1.1.4.1.1 "><p id="p2798819113819"><a name="p2798819113819"></a><a name="p2798819113819"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="12.6%" headers="mcps1.1.4.1.2 "><p id="p1988210327389"><a name="p1988210327389"></a><a name="p1988210327389"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.61%" headers="mcps1.1.4.1.3 "><p id="p17981419103810"><a name="p17981419103810"></a><a name="p17981419103810"></a>张量格式，默认为FORMAT_ND。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="11.06%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.59%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="75.35%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.59%" headers="mcps1.1.4.1.2 "><p id="p1157618141266"><a name="p1157618141266"></a><a name="p1157618141266"></a>EsCTensorHolder</p>
</td>
<td class="cellrowborder" valign="top" width="75.35%" headers="mcps1.1.4.1.3 "><p id="p95757141768"><a name="p95757141768"></a><a name="p95757141768"></a>返回创建的Const的张量持有者算子，失败时返回nullptr。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

## 调用示例<a name="section16305113853313"></a>

```
EsCGraphBuilder *graph = EsCreateGraphBuilder("graph_name");
std::vector<int64_t> data = {1, 2, 3};
std::vector<int64_t> dims = {3};
auto const_tensor = ge::es::EsCreateConst<int64_t>(graph, data.data(), dims.data(), dims.size(), ge::DT_INT64);
```

