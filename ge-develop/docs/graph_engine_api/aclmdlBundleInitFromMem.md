# aclmdlBundleInitFromMem<a name="ZH-CN_TOPIC_0000002528407301"></a>

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

在模型执行阶段，如果涉及动态更新变量的场景，可以调用本接口从内存中初始化模型。

## 函数原型<a name="section2067518173415"></a>

```
aclError aclmdlBundleInitFromMem(const void* model, size_t modelSize, void *varWeightPtr, size_t varWeightSize, uint32_t *bundleId)
```

## 参数说明<a name="section95959983419"></a>

<a name="zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.969999999999999%" id="mcps1.1.4.1.2"><p id="p1769255516412"><a name="p1769255516412"></a><a name="p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68.03%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0122830089_p173881843143514"><a name="zh-cn_topic_0122830089_p173881843143514"></a><a name="zh-cn_topic_0122830089_p173881843143514"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1919192774810"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p196693291619"><a name="p196693291619"></a><a name="p196693291619"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p76691522165"><a name="p76691522165"></a><a name="p76691522165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p58391552478"><a name="p58391552478"></a><a name="p58391552478"></a>存放模型数据的内存地址指针，由用户自行管理，加载aclmdlBundleLoadModel过程中不能释放该内存。</p>
<p id="p4388859358"><a name="p4388859358"></a><a name="p4388859358"></a>此处的模型文件是基于构图接口构建出来的，调用aclgrphBundleBuildModel接口编译模型、调用aclgrphBundleSaveModel接口保存模型，再由用户自行将保存出来的om模型文件读入内存，构图接口详细描述参见<span id="ph79081379286"><a name="ph79081379286"></a><a name="ph79081379286"></a>《图模式开发指南》</span>。</p>
<p id="p1580512422221"><a name="p1580512422221"></a><a name="p1580512422221"></a>应用运行在Host时，此处需申请Host上的内存；应用运行在Device时，此处需申请Device上的内存。内存申请接口请参见<a href="zh-cn_topic_0000001312400661.md">内存管理</a>。</p>
</td>
</tr>
<tr id="row856311577131"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p966710213162"><a name="p966710213162"></a><a name="p966710213162"></a>modelSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p6955195335220"><a name="p6955195335220"></a><a name="p6955195335220"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p64351912165911"><a name="p64351912165911"></a><a name="p64351912165911"></a>内存中的模型数据长度，单位Byte。</p>
</td>
</tr>
<tr id="row18987133142614"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p14642511012"><a name="p14642511012"></a><a name="p14642511012"></a>varWeightPtr</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p464195111015"><a name="p464195111015"></a><a name="p464195111015"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p84291942201119"><a name="p84291942201119"></a><a name="p84291942201119"></a>模型所需的可刷新权重内存的地址指针，由用户自行管理，模型执行过程中不能释放该内存。</p>
<p id="p1748275311918"><a name="p1748275311918"></a><a name="p1748275311918"></a>如果在此处传入空指针，表示由系统管理内存。</p>
</td>
</tr>
<tr id="row1952131351019"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p15521201311105"><a name="p15521201311105"></a><a name="p15521201311105"></a>varWeightSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p45213138104"><a name="p45213138104"></a><a name="p45213138104"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p879412295547"><a name="p879412295547"></a><a name="p879412295547"></a>模型所需的可刷新权重内存的大小，单位Byte。workPtr为空指针时无效。</p>
</td>
</tr>
<tr id="row7180892104"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1618111911017"><a name="p1618111911017"></a><a name="p1618111911017"></a>bundleId</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p1318116912105"><a name="p1318116912105"></a><a name="p1318116912105"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p11812971011"><a name="p11812971011"></a><a name="p11812971011"></a>系统成功初始化捆绑模型后，返回bundleId作为后续操作时识别模型的标志。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section16131193591515"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

