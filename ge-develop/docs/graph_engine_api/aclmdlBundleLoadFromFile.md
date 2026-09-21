# aclmdlBundleLoadFromFile<a name="ZH-CN_TOPIC_0000002006025225"></a>

## 产品支持情况<a name="section16107182283615"></a>

<a name="zh-cn_topic_0000002219420921_table14931115524110"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002219420921_row1993118556414"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002219420921_p29315553419"><a name="zh-cn_topic_0000002219420921_p29315553419"></a><a name="zh-cn_topic_0000002219420921_p29315553419"></a><span id="zh-cn_topic_0000002219420921_ph59311455164119"><a name="zh-cn_topic_0000002219420921_ph59311455164119"></a><a name="zh-cn_topic_0000002219420921_ph59311455164119"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002219420921_p59313557417"><a name="zh-cn_topic_0000002219420921_p59313557417"></a><a name="zh-cn_topic_0000002219420921_p59313557417"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002219420921_row1693117553411"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p1493195513412"><a name="zh-cn_topic_0000002219420921_p1493195513412"></a><a name="zh-cn_topic_0000002219420921_p1493195513412"></a><span id="zh-cn_topic_0000002219420921_ph1093110555418"><a name="zh-cn_topic_0000002219420921_ph1093110555418"></a><a name="zh-cn_topic_0000002219420921_ph1093110555418"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p20931175524111"><a name="zh-cn_topic_0000002219420921_p20931175524111"></a><a name="zh-cn_topic_0000002219420921_p20931175524111"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002219420921_row199312559416"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002219420921_p0931555144119"><a name="zh-cn_topic_0000002219420921_p0931555144119"></a><a name="zh-cn_topic_0000002219420921_p0931555144119"></a><span id="zh-cn_topic_0000002219420921_ph1693115559411"><a name="zh-cn_topic_0000002219420921_ph1693115559411"></a><a name="zh-cn_topic_0000002219420921_ph1693115559411"></a><term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000002219420921_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002219420921_p129321955154117"><a name="zh-cn_topic_0000002219420921_p129321955154117"></a><a name="zh-cn_topic_0000002219420921_p129321955154117"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

模型执行阶段若涉及动态更新变量的场景，可调用本接口从文件加载离线模型数据（适配昇腾AI处理器的离线模型），由系统内部管理内存。

**本接口需与以下其它接口配合使用**，以便实现动态更新变量的目的，关键接口的调用流程如下：

1.  基于构图接口编译并保存模型，模型中包含多个图，例如推理图、变量初始化图、变量更新图等。

    此处是调用aclgrphBundleBuildModel接口编译模型、调用aclgrphBundleSaveModel接口保存模型，接口详细描述参见《图模式开发指南》。

2.  调用[aclmdlBundleLoadFromFile](aclmdlBundleLoadFromFile.md)或[aclmdlBundleLoadFromMem](aclmdlBundleLoadFromMem.md)接口加载模型。
3.  调用[aclmdlBundleGetModelId](aclmdlBundleGetModelId.md)接口获取多个图的ID。
4.  根据多个图的ID，分别调用模型执行接口（例如[aclmdlExecute](aclmdlExecute.md)）执行各个图。

    若涉及变量更新，则在执行变量更新图之前，先调用[aclmdlSetDatasetTensorDesc](aclmdlSetDatasetTensorDesc.md)接口设置图的tensor描述信息，再执行变量更新图，然后再执行一次推理图。

5.  推理结束后，调用[aclmdlBundleUnload](aclmdlBundleUnload.md)接口卸载模型。

## 函数原型<a name="section2067518173415"></a>

```
aclError aclmdlBundleLoadFromFile(const char *modelPath, uint32_t *bundleId)
```

## 参数说明<a name="section95959983419"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1919192774810"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p196693291619"><a name="p196693291619"></a><a name="p196693291619"></a>modelPath</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p76691522165"><a name="p76691522165"></a><a name="p76691522165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p1120031519571"><a name="p1120031519571"></a><a name="p1120031519571"></a>模型文件路径的指针，路径中包含文件名。运行程序（APP）的用户需要对该存储路径有访问权限。</p>
<p id="p968205565312"><a name="p968205565312"></a><a name="p968205565312"></a>此处的模型文件是基于构图接口构建出来的，调用aclgrphBundleBuildModel接口编译模型、调用aclgrphBundleSaveModel接口保存模型。</p>
</td>
</tr>
<tr id="row18987133142614"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p966710213162"><a name="p966710213162"></a><a name="p966710213162"></a>bundleId</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p66671229166"><a name="p66671229166"></a><a name="p66671229166"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p8851114185610"><a name="p8851114185610"></a><a name="p8851114185610"></a>系统成功加载模型后，返回bundleId作为后续操作时识别模型的标志。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section16131193591515"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

