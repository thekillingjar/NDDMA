# aclmdlAddDatasetBuffer<a name="ZH-CN_TOPIC_0000001265241302"></a>

## 产品支持情况<a name="section16107182283615"></a>

<a name="zh-cn_topic_0000002219420921_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002219420921_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002219420921_p1883113061818"><a name="zh-cn_topic_0000002219420921_p1883113061818"></a><a name="zh-cn_topic_0000002219420921_p1883113061818"></a><span id="zh-cn_topic_0000002219420921_ph20833205312295"><a name="zh-cn_topic_0000002219420921_ph20833205312295"></a><a name="zh-cn_topic_0000002219420921_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002219420921_p783113012187"><a name="zh-cn_topic_0000002219420921_p783113012187"></a><a name="zh-cn_topic_0000002219420921_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002219420921_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p48327011813"><a name="zh-cn_topic_0000002219420921_p48327011813"></a><a name="zh-cn_topic_0000002219420921_p48327011813"></a><span id="zh-cn_topic_0000002219420921_ph583230201815"><a name="zh-cn_topic_0000002219420921_ph583230201815"></a><a name="zh-cn_topic_0000002219420921_ph583230201815"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p7948163910184"><a name="zh-cn_topic_0000002219420921_p7948163910184"></a><a name="zh-cn_topic_0000002219420921_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002219420921_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p14832120181815"><a name="zh-cn_topic_0000002219420921_p14832120181815"></a><a name="zh-cn_topic_0000002219420921_p14832120181815"></a><span id="zh-cn_topic_0000002219420921_ph1483216010188"><a name="zh-cn_topic_0000002219420921_ph1483216010188"></a><a name="zh-cn_topic_0000002219420921_ph1483216010188"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p19948143911820"><a name="zh-cn_topic_0000002219420921_p19948143911820"></a><a name="zh-cn_topic_0000002219420921_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section0117164520431"></a>

向aclmdlDataset中增加aclDataBuffer。

## 函数原型<a name="section133154911438"></a>

```
aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer)
```

## 参数说明<a name="section13616184164416"></a>

<a name="table133342381612"></a>
<table><thead align="left"><tr id="row173342038467"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p1633411386618"><a name="p1633411386618"></a><a name="p1633411386618"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p8334163811620"><a name="p8334163811620"></a><a name="p8334163811620"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p03354381462"><a name="p03354381462"></a><a name="p03354381462"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row12335193818617"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p73353381668"><a name="p73353381668"></a><a name="p73353381668"></a>dataset</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p8335203817611"><a name="p8335203817611"></a><a name="p8335203817611"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10329145511613"><a name="p10329145511613"></a><a name="p10329145511613"></a>待增加aclDataBuffer的aclmdlDataset地址指针。</p>
<p id="p093841194213"><a name="p093841194213"></a><a name="p093841194213"></a>需提前调用<a href="aclmdlCreateDataset.md">aclmdlCreateDataset</a>接口创建aclmdlDataset类型的数据。</p>
</td>
</tr>
<tr id="row758196578"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p205821261577"><a name="p205821261577"></a><a name="p205821261577"></a>dataBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1258266571"><a name="p1258266571"></a><a name="p1258266571"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p2582561173"><a name="p2582561173"></a><a name="p2582561173"></a>待增加的aclDataBuffer地址指针。</p>
<p id="p11154192018422"><a name="p11154192018422"></a><a name="p11154192018422"></a>需提前调用<span id="ph0151456133520"><a name="ph0151456133520"></a><a name="ph0151456133520"></a>aclCreateDataBuffer</span>接口创建aclDataBuffer类型的数据。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section162895447"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

