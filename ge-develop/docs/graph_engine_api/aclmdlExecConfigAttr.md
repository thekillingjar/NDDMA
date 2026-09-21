# aclmdlExecConfigAttr<a name="ZH-CN_TOPIC_0000001599679189"></a>

```
typedef enum {
    ACL_MDL_STREAM_SYNC_TIMEOUT = 0,
    ACL_MDL_EVENT_SYNC_TIMEOUT,
    ACL_MDL_WORK_ADDR_PTR,
    ACL_MDL_WORK_SIZET,
    ACL_MDL_MPAIMID_SIZET,      /** param reserved */ 
    ACL_MDL_AICQOS_SIZET,       /** param reserved */ 
    ACL_MDL_AICOST_SIZET,       /** param reserved */ 
    ACL_MDL_MEC_TIMETHR_SIZET   /** param reserved */ 
} aclmdlExecConfigAttr;
```

**表 1**  枚举项说明

<a name="table154428882117"></a>
<table><thead align="left"><tr id="row15442178172115"><th class="cellrowborder" valign="top" width="29.7%" id="mcps1.2.3.1.1"><p id="p10442188112114"><a name="p10442188112114"></a><a name="p10442188112114"></a>枚举项</p>
</th>
<th class="cellrowborder" valign="top" width="70.3%" id="mcps1.2.3.1.2"><p id="p84426814219"><a name="p84426814219"></a><a name="p84426814219"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row134425813216"><td class="cellrowborder" valign="top" width="29.7%" headers="mcps1.2.3.1.1 "><p id="p6388354203514"><a name="p6388354203514"></a><a name="p6388354203514"></a>ACL_MDL_STREAM_SYNC_TIMEOUT</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.2.3.1.2 "><p id="p18387554123518"><a name="p18387554123518"></a><a name="p18387554123518"></a>在执行模型推理时控制Stream任务的超时时间。该属性值为INT32类型。</p>
<p id="p14163161111313"><a name="p14163161111313"></a><a name="p14163161111313"></a>取值说明如下：</p>
<a name="ul589318220132"></a><a name="ul589318220132"></a><ul id="ul589318220132"><li>-1：表示永久等待，默认永久等待。</li><li>&gt;0：配置具体的超时时间，单位是毫秒。</li></ul>
</td>
</tr>
<tr id="row444313892114"><td class="cellrowborder" valign="top" width="29.7%" headers="mcps1.2.3.1.1 "><p id="p638585423511"><a name="p638585423511"></a><a name="p638585423511"></a>ACL_MDL_EVENT_SYNC_TIMEOUT</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.2.3.1.2 "><p id="p183856546352"><a name="p183856546352"></a><a name="p183856546352"></a>在执行模型推理时控制Event任务的超时时间。该属性值为INT32类型。</p>
<p id="p15666105812359"><a name="p15666105812359"></a><a name="p15666105812359"></a>取值说明如下：</p>
<a name="ul10666165813510"></a><a name="ul10666165813510"></a><ul id="ul10666165813510"><li>-1：表示永久等待，默认永久等待。</li><li>&gt;0：配置具体的超时时间，单位是毫秒。</li></ul>
</td>
</tr>
<tr id="row1144315862113"><td class="cellrowborder" valign="top" width="29.7%" headers="mcps1.2.3.1.1 "><p id="p17377254143520"><a name="p17377254143520"></a><a name="p17377254143520"></a>ACL_MDL_WORK_ADDR_PTR</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.2.3.1.2 "><p id="p9149101561113"><a name="p9149101561113"></a><a name="p9149101561113"></a>模型所需工作内存（Device上存放模型执行过程中的临时数据）的指针，由用户管理工作内存。一般用于模型一次加载、多并发执行的场景。</p>
<p id="p3832144135111"><a name="p3832144135111"></a><a name="p3832144135111"></a>如果同时配置ACL_MDL_WORK_ADDR_PTR以及<span id="ph9100152092418"><a name="ph9100152092418"></a><a name="ph9100152092418"></a>aclrtStreamConfigAttr</span>中的ACL_RT_STREAM_WORK_ADDR_PTR（表示Stream上模型的工作内存），则以ACL_MDL_WORK_ADDR_PTR优先。</p>
<p id="p14790171914463"><a name="p14790171914463"></a><a name="p14790171914463"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row6443168192118"><td class="cellrowborder" valign="top" width="29.7%" headers="mcps1.2.3.1.1 "><p id="p73757545351"><a name="p73757545351"></a><a name="p73757545351"></a>ACL_MDL_WORK_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.2.3.1.2 "><p id="p53742540355"><a name="p53742540355"></a><a name="p53742540355"></a>模型所需工作内存的大小，单位为Byte。一般用于模型一次加载、多并发执行的场景。</p>
<p id="p18583218105817"><a name="p18583218105817"></a><a name="p18583218105817"></a>当前版本不支持该配置。</p>
</td>
</tr>
<tr id="row1032816324387"><td class="cellrowborder" valign="top" width="29.7%" headers="mcps1.2.3.1.1 "><p id="p1032819325387"><a name="p1032819325387"></a><a name="p1032819325387"></a>ACL_MDL_MPAIMID_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.2.3.1.2 "><p id="p10328103215387"><a name="p10328103215387"></a><a name="p10328103215387"></a>预留配置。</p>
</td>
</tr>
<tr id="row74431810213"><td class="cellrowborder" valign="top" width="29.7%" headers="mcps1.2.3.1.1 "><p id="p437345453510"><a name="p437345453510"></a><a name="p437345453510"></a>ACL_MDL_AICQOS_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.2.3.1.2 "><p id="p153721354103519"><a name="p153721354103519"></a><a name="p153721354103519"></a>预留配置。</p>
</td>
</tr>
<tr id="row6443882211"><td class="cellrowborder" valign="top" width="29.7%" headers="mcps1.2.3.1.1 "><p id="p237105453517"><a name="p237105453517"></a><a name="p237105453517"></a>ACL_MDL_AICOST_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.2.3.1.2 "><p id="p2037010544358"><a name="p2037010544358"></a><a name="p2037010544358"></a>预留配置。</p>
</td>
</tr>
<tr id="row84434816215"><td class="cellrowborder" valign="top" width="29.7%" headers="mcps1.2.3.1.1 "><p id="p12369115410356"><a name="p12369115410356"></a><a name="p12369115410356"></a>ACL_MDL_MEC_TIMETHR_SIZET</p>
</td>
<td class="cellrowborder" valign="top" width="70.3%" headers="mcps1.2.3.1.2 "><p id="p14368125453510"><a name="p14368125453510"></a><a name="p14368125453510"></a>预留配置。</p>
</td>
</tr>
</tbody>
</table>

