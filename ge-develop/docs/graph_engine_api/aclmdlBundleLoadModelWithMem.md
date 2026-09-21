# aclmdlBundleLoadModelWithMem<a name="ZH-CN_TOPIC_0000002496887302"></a>

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

根据图索引加载模型中的图，由用户自行管理模型运行的内存。

## 函数原型<a name="section2067518173415"></a>

```
aclError aclmdlBundleLoadModelWithMem(uint32_t bundleId, size_t index, void *workPtr, size_t workSize, void *weightPtr, size_t weightSize, uint32_t *modelId)
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
<tbody><tr id="row1919192774810"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p196693291619"><a name="p196693291619"></a><a name="p196693291619"></a>bundleId</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p76691522165"><a name="p76691522165"></a><a name="p76691522165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p9796119191019"><a name="p9796119191019"></a><a name="p9796119191019"></a>通过<a href="aclmdlBundleInitFromFile.md">aclmdlBundleInitFromFile</a>或者<a href="aclmdlBundleInitFromMem.md">aclmdlBundleInitFromMem</a>接口初始化模型成功后返回的bundleId。</p>
</td>
</tr>
<tr id="row856311577131"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p12526236536"><a name="p12526236536"></a><a name="p12526236536"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p85264361031"><a name="p85264361031"></a><a name="p85264361031"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p866642171612"><a name="p866642171612"></a><a name="p866642171612"></a>索引。</p>
<p id="p5103103751315"><a name="p5103103751315"></a><a name="p5103103751315"></a>用户调用<a href="aclmdlBundleGetQueryModelNum.md">aclmdlBundleGetQueryModelNum</a>接口获取图总数后，这个index的取值范围：[0, (图总数-1)]。</p>
</td>
</tr>
<tr id="row1295194113201"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p13798191516347"><a name="p13798191516347"></a><a name="p13798191516347"></a>workPtr</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p17798815103410"><a name="p17798815103410"></a><a name="p17798815103410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p2798415163418"><a name="p2798415163418"></a><a name="p2798415163418"></a>Device上模型所需工作内存（存放模型执行过程中的临时数据）的地址指针，由用户自行管理，模型执行过程中不能释放该内存。</p>
<p id="p1633916554719"><a name="p1633916554719"></a><a name="p1633916554719"></a>如果在workPtr参数处传入空指针，表示由系统管理内存。</p>
<div class="p" id="p385084311181"><a name="p385084311181"></a><a name="p385084311181"></a>由用户自行管理工作内存时，如果多个模型串行执行，支持共用同一个工作内存，但用户需确保模型的串行执行顺序、且工作内存的大小需按多个模型中最大工作内存的大小来申请，例如通过以下方式保证串行：<a name="zh-cn_topic_0000001312481845_ul1417911224180"></a><a name="zh-cn_topic_0000001312481845_ul1417911224180"></a><ul id="zh-cn_topic_0000001312481845_ul1417911224180"><li>同步模型执行时，加锁，保证执行任务串行。</li><li>异步模型执行时，使用同一个stream，保证执行任务串行。</li></ul>
</div>
</td>
</tr>
<tr id="row17469575203"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1804171893413"><a name="p1804171893413"></a><a name="p1804171893413"></a>workSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p680421813414"><a name="p680421813414"></a><a name="p680421813414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p2804131811342"><a name="p2804131811342"></a><a name="p2804131811342"></a>模型所需工作内存的大小，单位Byte。workPtr为空指针时无效。</p>
</td>
</tr>
<tr id="row4137125312019"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1786693715015"><a name="p1786693715015"></a><a name="p1786693715015"></a>weightPtr</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p188661037901"><a name="p188661037901"></a><a name="p188661037901"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p1286619377013"><a name="p1286619377013"></a><a name="p1286619377013"></a>Device上模型权值内存（存放权值数据）的地址指针，由用户自行管理，模型执行过程中不能释放该内存。</p>
<p id="p17628152318912"><a name="p17628152318912"></a><a name="p17628152318912"></a>如果在weightPtr参数处传入空指针，表示由系统管理内存。</p>
<p id="p83148919195"><a name="p83148919195"></a><a name="p83148919195"></a>由用户自行管理权值内存时，在多线程场景下，对于同一个模型，如果在每个线程中都加载了一次，支持共用weightPtr，因为weightPtr内存在推理过程中是只读的。此处需注意，在共用weightPtr期间，不能释放weightPtr。</p>
</td>
</tr>
<tr id="row169291645122017"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p178343395019"><a name="p178343395019"></a><a name="p178343395019"></a>weightSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p483414396016"><a name="p483414396016"></a><a name="p483414396016"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p28342391501"><a name="p28342391501"></a><a name="p28342391501"></a>模型所需权值内存的大小，单位Byte。weightPtr为空指针时无效。</p>
</td>
</tr>
<tr id="row18987133142614"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p1978812145175"><a name="p1978812145175"></a><a name="p1978812145175"></a>modelId</p>
</td>
<td class="cellrowborder" valign="top" width="13.969999999999999%" headers="mcps1.1.4.1.2 "><p id="p3788201441714"><a name="p3788201441714"></a><a name="p3788201441714"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.1.4.1.3 "><p id="p1178731471714"><a name="p1178731471714"></a><a name="p1178731471714"></a>实际可执行的图ID。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section16131193591515"></a>

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

