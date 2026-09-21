# aclmdlGetDatasetNumBuffers<a name="ZH-CN_TOPIC_0000001264921746"></a>

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

获取aclmdlDataset中aclDataBuffer的个数。

## 函数原型<a name="section133154911438"></a>

```
size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataset)
```

## 参数说明<a name="section13616184164416"></a>

<a name="table1527919256810"></a>
<table><thead align="left"><tr id="row227932516813"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p132793252810"><a name="p132793252810"></a><a name="p132793252810"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p15279225883"><a name="p15279225883"></a><a name="p15279225883"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p1627918251083"><a name="p1627918251083"></a><a name="p1627918251083"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row627902516814"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p62799251187"><a name="p62799251187"></a><a name="p62799251187"></a>dataset</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p328019251384"><a name="p328019251384"></a><a name="p328019251384"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p10280202514819"><a name="p10280202514819"></a><a name="p10280202514819"></a>aclmdlDataset类型的指针。</p>
<p id="p3119530134311"><a name="p3119530134311"></a><a name="p3119530134311"></a>需提前调用<a href="aclmdlCreateDataset.md">aclmdlCreateDataset</a>接口创建aclmdlDataset类型的数据。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section162895447"></a>

aclDataBuffer的个数。

