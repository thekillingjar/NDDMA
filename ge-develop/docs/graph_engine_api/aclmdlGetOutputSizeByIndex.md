# aclmdlGetOutputSizeByIndex<a name="ZH-CN_TOPIC_0000001265081410"></a>

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

## 功能说明<a name="section111511450711"></a>

根据模型描述信息获取指定输出的大小，单位为Byte。

## 函数原型<a name="section119932012682"></a>

```
size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index)
```

## 参数说明<a name="section135141717811"></a>

<a name="table1831221142"></a>
<table><thead align="left"><tr id="row3332181419"><th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.1"><p id="p11372191420"><a name="p11372191420"></a><a name="p11372191420"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="14.000000000000002%" id="mcps1.1.4.1.2"><p id="p13320291417"><a name="p13320291417"></a><a name="p13320291417"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68%" id="mcps1.1.4.1.3"><p id="p1940213149"><a name="p1940213149"></a><a name="p1940213149"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row134122191416"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p14132111415"><a name="p14132111415"></a><a name="p14132111415"></a>modelDesc</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p144122141415"><a name="p144122141415"></a><a name="p144122141415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p4414215143"><a name="p4414215143"></a><a name="p4414215143"></a>aclmdlDesc类型的指针。</p>
<p id="p35754134454"><a name="p35754134454"></a><a name="p35754134454"></a>需提前调用<a href="aclmdlCreateDesc.md">aclmdlCreateDesc</a>接口创建aclmdlDesc类型的数据。</p>
</td>
</tr>
<tr id="row218712903311"><td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.1 "><p id="p3188029203315"><a name="p3188029203315"></a><a name="p3188029203315"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="14.000000000000002%" headers="mcps1.1.4.1.2 "><p id="p9188829183314"><a name="p9188829183314"></a><a name="p9188829183314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68%" headers="mcps1.1.4.1.3 "><p id="p131881029193318"><a name="p131881029193318"></a><a name="p131881029193318"></a>指定获取第几个输出的大小，index值从0开始。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section184151120582"></a>

针对动态Batch、动态分辨率（宽高）的场景，返回最大档位对应的输出的大小；静态场景下，返回指定输出的大小。单位是Byte。

## 约束说明<a name="section44073182283"></a>

如果通过本接口获取的大小为0，有可能是由于输出Shape的范围不确定，当前支持以下两种处理方式：

-   方式一：**系统内部自行申请对应index的输出内存**，节省内存，但内存数据使用结束后，需由用户释放内存，同时，系统内部申请内存时涉及内存拷贝，可能涉及性能损耗。该方式仅支持在使用[aclmdlExecute](aclmdlExecute.md)、[aclmdlExecuteV2](aclmdlExecuteV2.md)推理接口时使用。

    调用aclCreateDataBuffer接口创建存放对应index**输出数据**的aclDataBuffer类型时，**支持在data参数处传入nullptr，同时size需设置为0**，表示创建一个空的aclDataBuffer类型，然后在模型执行过程中，系统**内部自行计算并申请**该index输出的内存。使用该方式可节省内存，但内存数据使用结束后，需由用户释放内存并重置aclDataBuffer，同时，系统内部申请内存时涉及内存拷贝，可能涉及性能损耗。

    释放内存并重置aclDataBuffer的示例代码如下：

    ```
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output, 0); // 根据index获取对应的dataBuffer        
    void *data = aclGetDataBufferAddr(dataBuffer);  // 获取data的Device指针
    aclrtFree(data ); // 释放Device内存
    aclUpdateDataBuffer(dataBuffer, nullptr, 0); // 重置dataBuffer里面内容，以便下次推理
    ```

-   方式二：**用户预估输出内存大小，并申请内存**，由用户自行管理内存，但内存大小可能不够或超出，不够时系统会校验报错，超出时会浪费内存。

    用户需先根据实际情况预估一块较大的输出内存，在模型执行过程中，系统会校验用户指定的输出内存大小是否符合要求，如果不符合要求，系统会返回报错，并在报错信息中提示具体需要多大的输出内存。您可以通过以下两种方式查看报错：

    -   获取应用类日志，查看ERROR级别的报错，如何获取并查看日志请参见《日志参考》。
    -   在应用程序中调用aclGetRecentErrMsg接口获取报错。

