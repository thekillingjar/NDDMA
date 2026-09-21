# aclSetTensorFormat<a name="ZH-CN_TOPIC_0000001265081914"></a>

## 产品支持情况<a name="section8178181118225"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p7948163910184"><a name="p7948163910184"></a><a name="p7948163910184"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section1123611141553"></a>

调用[aclCreateTensorDesc](aclCreateTensorDesc.md)接口创建tensor描述信息后，可通过aclSetTensorFormat设置昇腾AI处理器内部处理tensor时的Format信息。

例如，某算子的原始Format为NHWC（原始Shape为4维）或者NDHWC（原始Shape为5维），为了提高数据访问效率，在进行模型转换或者在线构建网络时会自动转换Format，将原始数据Format转化为昇腾AI处理器内部计算数据时用的Format（NC1HWC0或者NDC1HWC0），同时进行推导Shape用于内部计算。

## 函数原型<a name="section1696391717558"></a>

```
aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format)
```

## 参数说明<a name="section129037230558"></a>

<a name="table15185939162113"></a>
<table><thead align="left"><tr id="row818513911216"><th class="cellrowborder" valign="top" width="20.52%" id="mcps1.1.4.1.1"><p id="p20185103918218"><a name="p20185103918218"></a><a name="p20185103918218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.52%" id="mcps1.1.4.1.2"><p id="p11185143916215"><a name="p11185143916215"></a><a name="p11185143916215"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68.96%" id="mcps1.1.4.1.3"><p id="p191851939172115"><a name="p191851939172115"></a><a name="p191851939172115"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row2608947113210"><td class="cellrowborder" valign="top" width="20.52%" headers="mcps1.1.4.1.1 "><p id="p3608647113219"><a name="p3608647113219"></a><a name="p3608647113219"></a>desc</p>
</td>
<td class="cellrowborder" valign="top" width="10.52%" headers="mcps1.1.4.1.2 "><p id="p12608447173219"><a name="p12608447173219"></a><a name="p12608447173219"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.96%" headers="mcps1.1.4.1.3 "><p id="p2648133315150"><a name="p2648133315150"></a><a name="p2648133315150"></a>aclTensorDesc类型的指针。</p>
<p id="p1861313125319"><a name="p1861313125319"></a><a name="p1861313125319"></a>需提前调用<a href="aclCreateTensorDesc.md">aclCreateTensorDesc</a>接口创建aclTensorDesc类型。</p>
</td>
</tr>
<tr id="row1765514213210"><td class="cellrowborder" valign="top" width="20.52%" headers="mcps1.1.4.1.1 "><p id="p1565604213211"><a name="p1565604213211"></a><a name="p1565604213211"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="10.52%" headers="mcps1.1.4.1.2 "><p id="p265674220325"><a name="p265674220325"></a><a name="p265674220325"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.96%" headers="mcps1.1.4.1.3 "><p id="p06561642103217"><a name="p06561642103217"></a><a name="p06561642103217"></a>需要设置的format。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section197343339559"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

