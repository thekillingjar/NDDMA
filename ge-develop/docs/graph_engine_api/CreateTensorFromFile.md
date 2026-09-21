# CreateTensorFromFile<a name="ZH-CN_TOPIC_0000002486895312"></a>

## 功能说明<a name="section44282627"></a>

从binary文件创建指定数据类型和张量形状的张量。

## 函数原型<a name="section1831611148519"></a>

```
template <typename T>
std::unique_ptr<Tensor> CreateTensorFromFile(const char *data_file_path, const std::vector<int64_t> &dims, DataType dt, Format format = FORMAT_ND);
```

## 参数说明<a name="section62999330"></a>

<a name="table10309404"></a>
<table><thead align="left"><tr id="row47530006"><th class="cellrowborder" valign="top" width="16.18%" id="mcps1.1.4.1.1"><p id="p24725298"><a name="p24725298"></a><a name="p24725298"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.35%" id="mcps1.1.4.1.2"><p id="p56592155"><a name="p56592155"></a><a name="p56592155"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.47%" id="mcps1.1.4.1.3"><p id="p54897010"><a name="p54897010"></a><a name="p54897010"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row17472816"><td class="cellrowborder" valign="top" width="16.18%" headers="mcps1.1.4.1.1 "><p id="p1583161612618"><a name="p1583161612618"></a><a name="p1583161612618"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.2 "><p id="p165822162612"><a name="p165822162612"></a><a name="p165822162612"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.47%" headers="mcps1.1.4.1.3 "><p id="p185820162619"><a name="p185820162619"></a><a name="p185820162619"></a>张量数据类型。</p>
</td>
</tr>
<tr id="row107217586567"><td class="cellrowborder" valign="top" width="16.18%" headers="mcps1.1.4.1.1 "><p id="p35821516364"><a name="p35821516364"></a><a name="p35821516364"></a>data_file_path</p>
</td>
<td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.2 "><p id="p1658219161461"><a name="p1658219161461"></a><a name="p1658219161461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.47%" headers="mcps1.1.4.1.3 "><p id="p11581191613619"><a name="p11581191613619"></a><a name="p11581191613619"></a>binary文件数据路径。</p>
</td>
</tr>
<tr id="row821633613614"><td class="cellrowborder" valign="top" width="16.18%" headers="mcps1.1.4.1.1 "><p id="p15811716166"><a name="p15811716166"></a><a name="p15811716166"></a>dims</p>
</td>
<td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.2 "><p id="p1158111161162"><a name="p1158111161162"></a><a name="p1158111161162"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.47%" headers="mcps1.1.4.1.3 "><p id="p25811516365"><a name="p25811516365"></a><a name="p25811516365"></a>张量维度向量。</p>
</td>
</tr>
<tr id="row7269125213513"><td class="cellrowborder" valign="top" width="16.18%" headers="mcps1.1.4.1.1 "><p id="p826945293513"><a name="p826945293513"></a><a name="p826945293513"></a>dt</p>
</td>
<td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.2 "><p id="p12269155243510"><a name="p12269155243510"></a><a name="p12269155243510"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.47%" headers="mcps1.1.4.1.3 "><p id="p6269135263517"><a name="p6269135263517"></a><a name="p6269135263517"></a>张量的数据类型。</p>
</td>
</tr>
<tr id="row139916589351"><td class="cellrowborder" valign="top" width="16.18%" headers="mcps1.1.4.1.1 "><p id="p43991583356"><a name="p43991583356"></a><a name="p43991583356"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.2 "><p id="p153991858113511"><a name="p153991858113511"></a><a name="p153991858113511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.47%" headers="mcps1.1.4.1.3 "><p id="p16399358173518"><a name="p16399358173518"></a><a name="p16399358173518"></a>张量格式，默认为FORMAT_ND。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<a name="table2601186"></a>
<table><thead align="left"><tr id="row1832323"><th class="cellrowborder" valign="top" width="9.75%" id="mcps1.1.4.1.1"><p id="p14200498"><a name="p14200498"></a><a name="p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="16.08%" id="mcps1.1.4.1.2"><p id="p9389685"><a name="p9389685"></a><a name="p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="74.17%" id="mcps1.1.4.1.3"><p id="p22367029"><a name="p22367029"></a><a name="p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row66898905"><td class="cellrowborder" valign="top" width="9.75%" headers="mcps1.1.4.1.1 "><p id="p50102218"><a name="p50102218"></a><a name="p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="16.08%" headers="mcps1.1.4.1.2 "><p id="p1157618141266"><a name="p1157618141266"></a><a name="p1157618141266"></a>std::unique_ptr&lt;Tensor&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="74.17%" headers="mcps1.1.4.1.3 "><p id="p95757141768"><a name="p95757141768"></a><a name="p95757141768"></a>返回创建的张量智能指针，失败时返回nullptr。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

无

## 调用示例<a name="section16305113853313"></a>

```
std::vector<int64_t> dims = {1};
auto tensor = CreateTensorFromFile<int64_t>(file_path.c_str(), dims, ge::DT_INT64, ge::FORMAT_ALL);
```

