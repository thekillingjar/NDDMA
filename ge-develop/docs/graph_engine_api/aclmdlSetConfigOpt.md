# aclmdlSetConfigOpt<a name="ZH-CN_TOPIC_0000001312721821"></a>

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

## 功能说明<a name="section259105813316"></a>

设置模型加载的配置对象中的各属性的取值，包括模型执行的优先级、模型的文件路径或内存地址、内存大小等。

**本接口需要与以下其它接口配合**，实现模型加载功能：

1.  调用[aclmdlCreateConfigHandle](aclmdlCreateConfigHandle.md)接口创建模型加载的配置对象。
2.  （可选）调用[aclmdlSetExternalWeightAddress](aclmdlSetExternalWeightAddress.md)接口配置存放外置权重的Device内存。
3.  多次调用[aclmdlSetConfigOpt](aclmdlSetConfigOpt.md)接口设置配置对象中每个属性的值。
4.  调用[aclmdlLoadWithConfig](aclmdlLoadWithConfig.md)接口指定模型加载时需要的配置信息，并进行模型加载。
5.  模型加载成功后，调用[aclmdlDestroyConfigHandle](aclmdlDestroyConfigHandle.md)接口销毁。

## 函数原型<a name="section2067518173415"></a>

```
aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr, const void *attrValue, size_t valueSize)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1919192774810"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p15161451803"><a name="p15161451803"></a><a name="p15161451803"></a>handle</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p115114513010"><a name="p115114513010"></a><a name="p115114513010"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p13605195465711"><a name="p13605195465711"></a><a name="p13605195465711"></a>模型加载的配置对象的指针。需提前调用<a href="aclmdlCreateConfigHandle.md">aclmdlCreateConfigHandle</a>接口创建该对象。</p>
</td>
</tr>
<tr id="row18987133142614"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p598883182618"><a name="p598883182618"></a><a name="p598883182618"></a>attr</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p14988938265"><a name="p14988938265"></a><a name="p14988938265"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p159885382612"><a name="p159885382612"></a><a name="p159885382612"></a>指定需设置的属性。</p>
</td>
</tr>
<tr id="row617331362615"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p51732013102614"><a name="p51732013102614"></a><a name="p51732013102614"></a>attrValue</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p1217331362617"><a name="p1217331362617"></a><a name="p1217331362617"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p41731213142610"><a name="p41731213142610"></a><a name="p41731213142610"></a>指向属性值的指针，attr对应的属性取值。</p>
<p id="p10451181712146"><a name="p10451181712146"></a><a name="p10451181712146"></a>如果属性值本身是指针，则传入该指针的地址。</p>
</td>
</tr>
<tr id="row18728717152617"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p37291917112616"><a name="p37291917112616"></a><a name="p37291917112616"></a>valueSize</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p10729217132618"><a name="p10729217132618"></a><a name="p10729217132618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p17729417122614"><a name="p17729417122614"></a><a name="p17729417122614"></a>attrValue部分的数据长度。</p>
<p id="p15101194111244"><a name="p15101194111244"></a><a name="p15101194111244"></a>用户可使用C/C++标准库的函数sizeof(*attrValue)查询数据长度。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section15770391345"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 参考资源<a name="section1035824101614"></a>

使用[aclmdlSetConfigOpt](aclmdlSetConfigOpt.md)接口、[aclmdlLoadWithConfig](aclmdlLoadWithConfig.md)接口时，是通过配置对象中的属性来区分，在加载模型时是从文件加载，还是从内存加载，以及内存是由系统内部管理，还是由用户管理。

当前还提供了以下接口实现模型加载的功能，从使用的接口上区分从文件加载，还是从内存加载，以及内存是由系统内部管理，还是由用户管理。

-   [aclmdlLoadFromFile](aclmdlLoadFromFile.md)接口
-   [aclmdlLoadFromMem](aclmdlLoadFromMem.md)接口
-   [aclmdlLoadFromFileWithMem](aclmdlLoadFromFileWithMem.md)接口
-   [aclmdlLoadFromMemWithMem](aclmdlLoadFromMemWithMem.md)接口
-   [aclmdlLoadFromFileWithQ](aclmdlLoadFromFileWithQ.md)接口
-   [aclmdlLoadFromMemWithQ](aclmdlLoadFromMemWithQ.md)接口

