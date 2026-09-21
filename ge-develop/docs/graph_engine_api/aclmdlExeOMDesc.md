# aclmdlExeOMDesc<a name="ZH-CN_TOPIC_0000001877350049"></a>

```
typedef struct aclmdlExeOMDesc {
    size_t workSize;
    size_t weightSize;
    size_t modelDescSize;
    size_t kernelSize;
    size_t kernelArgsSize;
    size_t staticTaskSize;
    size_t dynamicTaskSize;
    size_t fifoTaskSize;
    size_t reserved[8];
} aclmdlExeOMDesc;
```

<a name="zh-cn_topic_0249624707_table6284194414136"></a>
<table><thead align="left"><tr id="zh-cn_topic_0249624707_row341484411134"><th class="cellrowborder" valign="top" width="20.9%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0249624707_p154141244121314"><a name="zh-cn_topic_0249624707_p154141244121314"></a><a name="zh-cn_topic_0249624707_p154141244121314"></a>成员名称</p>
</th>
<th class="cellrowborder" valign="top" width="79.10000000000001%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0249624707_p10414344151315"><a name="zh-cn_topic_0249624707_p10414344151315"></a><a name="zh-cn_topic_0249624707_p10414344151315"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0249624707_row754710296481"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p106121425182514"><a name="p106121425182514"></a><a name="p106121425182514"></a>workSize</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p17862221817"><a name="p17862221817"></a><a name="p17862221817"></a>模型执行时所需的工作内存的大小，单位Byte。</p>
</td>
</tr>
<tr id="zh-cn_topic_0249624707_row936773214820"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p0611152513258"><a name="p0611152513258"></a><a name="p0611152513258"></a>weightSize</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p46482404189"><a name="p46482404189"></a><a name="p46482404189"></a>模型执行时所需的权值内存的大小，单位Byte。</p>
</td>
</tr>
<tr id="zh-cn_topic_0249624707_row194149449133"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p861112515256"><a name="p861112515256"></a><a name="p861112515256"></a>modelDescSize</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p176101325162514"><a name="p176101325162514"></a><a name="p176101325162514"></a>存放模型描述信息所需的内存大小，单位Byte。</p>
</td>
</tr>
<tr id="zh-cn_topic_0249624707_row141411445133"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p1661017259251"><a name="p1661017259251"></a><a name="p1661017259251"></a>kernelSize</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p9609225172516"><a name="p9609225172516"></a><a name="p9609225172516"></a>存放TBE算子kernel（算子的*.o与*.json）所需的内存大小，单位Byte。</p>
</td>
</tr>
<tr id="row52491030142513"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p824918308256"><a name="p824918308256"></a><a name="p824918308256"></a>kernelArgsSize</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p724963032514"><a name="p724963032514"></a><a name="p724963032514"></a>存放TBE算子kernel参数所需的内存大小，单位Byte。</p>
</td>
</tr>
<tr id="row9893183117257"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p8893231172517"><a name="p8893231172517"></a><a name="p8893231172517"></a>staticTaskSize</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p1089313114257"><a name="p1089313114257"></a><a name="p1089313114257"></a>存放静态shape任务描述信息所需的内存大小，单位Byte。</p>
</td>
</tr>
<tr id="row13511152832511"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p1651182832511"><a name="p1651182832511"></a><a name="p1651182832511"></a>dynamicTaskSize</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p151162815253"><a name="p151162815253"></a><a name="p151162815253"></a>存放动态shape任务描述信息所需的内存大小，单位Byte。</p>
</td>
</tr>
<tr id="row362475916247"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p176243598240"><a name="p176243598240"></a><a name="p176243598240"></a>fifoTaskSize</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p16624959162418"><a name="p16624959162418"></a><a name="p16624959162418"></a>存放模型级别的全局内存大小，单位Byte。</p>
<p id="p5422144311401"><a name="p5422144311401"></a><a name="p5422144311401"></a>若某个模型在推理时，其每一层的输入来自上一层的输出以及前面几轮推理结果拼接而成时，则需使用模型级别的全局内存将该模型所需的输入数据保存下来，供后续推理使用。</p>
</td>
</tr>
<tr id="zh-cn_topic_0249624707_row941584421312"><td class="cellrowborder" valign="top" width="20.9%" headers="mcps1.1.3.1.1 "><p id="p196092025132519"><a name="p196092025132519"></a><a name="p196092025132519"></a>reserved</p>
</td>
<td class="cellrowborder" valign="top" width="79.10000000000001%" headers="mcps1.1.3.1.2 "><p id="p460842516259"><a name="p460842516259"></a><a name="p460842516259"></a>预留值。</p>
</td>
</tr>
</tbody>
</table>

