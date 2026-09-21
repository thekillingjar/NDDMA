# aclgrphProfCreateConfig<a name="ZH-CN_TOPIC_0000001265085950"></a>

## 产品支持情况<a name="section1993214622115"></a>

<a name="zh-cn_topic_0000001265245842_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265245842_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001265245842_p1883113061818"><a name="zh-cn_topic_0000001265245842_p1883113061818"></a><a name="zh-cn_topic_0000001265245842_p1883113061818"></a><span id="zh-cn_topic_0000001265245842_ph20833205312295"><a name="zh-cn_topic_0000001265245842_ph20833205312295"></a><a name="zh-cn_topic_0000001265245842_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001265245842_p783113012187"><a name="zh-cn_topic_0000001265245842_p783113012187"></a><a name="zh-cn_topic_0000001265245842_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265245842_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265245842_p48327011813"><a name="zh-cn_topic_0000001265245842_p48327011813"></a><a name="zh-cn_topic_0000001265245842_p48327011813"></a><span id="zh-cn_topic_0000001265245842_ph583230201815"><a name="zh-cn_topic_0000001265245842_ph583230201815"></a><a name="zh-cn_topic_0000001265245842_ph583230201815"></a><term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265245842_p108715341013"><a name="zh-cn_topic_0000001265245842_p108715341013"></a><a name="zh-cn_topic_0000001265245842_p108715341013"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265245842_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265245842_p14832120181815"><a name="zh-cn_topic_0000001265245842_p14832120181815"></a><a name="zh-cn_topic_0000001265245842_p14832120181815"></a><span id="zh-cn_topic_0000001265245842_ph1483216010188"><a name="zh-cn_topic_0000001265245842_ph1483216010188"></a><a name="zh-cn_topic_0000001265245842_ph1483216010188"></a><term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001265245842_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265245842_p19948143911820"><a name="zh-cn_topic_0000001265245842_p19948143911820"></a><a name="zh-cn_topic_0000001265245842_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <ge/ge\_prof.h\>
-   库文件：libmsprofiler.so

## 功能说明<a name="section48327783"></a>

创建Profiling配置信息。

## 函数原型<a name="section32296866"></a>

```
aclgrphProfConfig *aclgrphProfCreateConfig(uint32_t *deviceid_list, uint32_t device_nums, ProfilingAicoreMetrics aicore_metrics, ProfAicoreEvents *aicore_events, uint64_t data_type_config)
```

## 参数说明<a name="section22236344"></a>

<a name="table41781154"></a>
<table><thead align="left"><tr id="row42915720"><th class="cellrowborder" valign="top" width="14.2%" id="mcps1.1.4.1.1"><p id="p53621294"><a name="p53621294"></a><a name="p53621294"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.950000000000001%" id="mcps1.1.4.1.2"><p id="p48357528"><a name="p48357528"></a><a name="p48357528"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.85%" id="mcps1.1.4.1.3"><p id="p24645678"><a name="p24645678"></a><a name="p24645678"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row50142938"><td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.1.4.1.1 "><p id="p35046145"><a name="p35046145"></a><a name="p35046145"></a>deviceid_list</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.1.4.1.2 "><p id="p20165517"><a name="p20165517"></a><a name="p20165517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.85%" headers="mcps1.1.4.1.3 "><p id="p22794214"><a name="p22794214"></a><a name="p22794214"></a>需要采集数据的Device ID列表。</p>
</td>
</tr>
<tr id="row3821339"><td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.1.4.1.1 "><p id="p41093009"><a name="p41093009"></a><a name="p41093009"></a>device_nums</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.1.4.1.2 "><p id="p40199432"><a name="p40199432"></a><a name="p40199432"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.85%" headers="mcps1.1.4.1.3 "><p id="p34928551"><a name="p34928551"></a><a name="p34928551"></a>Device个数。需要保证和deviceid_list中的Device数目一致。</p>
</td>
</tr>
<tr id="row45921510"><td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.1.4.1.1 "><p id="p28654805"><a name="p28654805"></a><a name="p28654805"></a>aicore_metrics</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.1.4.1.2 "><p id="p39337857"><a name="p39337857"></a><a name="p39337857"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.85%" headers="mcps1.1.4.1.3 "><p id="p32249809"><a name="p32249809"></a><a name="p32249809"></a>AI Core metrics，枚举值请参见<a href="ProfilingAicoreMetrics.md">ProfilingAicoreMetrics</a>。</p>
</td>
</tr>
<tr id="row15367454"><td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.1.4.1.1 "><p id="p36804285"><a name="p36804285"></a><a name="p36804285"></a>aicore_events</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.1.4.1.2 "><p id="p28357104"><a name="p28357104"></a><a name="p28357104"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.85%" headers="mcps1.1.4.1.3 "><p id="p15224096"><a name="p15224096"></a><a name="p15224096"></a>AI Core events，预留项。</p>
</td>
</tr>
<tr id="row2799138"><td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.1.4.1.1 "><p id="p25403607"><a name="p25403607"></a><a name="p25403607"></a>data_type_config</p>
</td>
<td class="cellrowborder" valign="top" width="9.950000000000001%" headers="mcps1.1.4.1.2 "><p id="p44426301"><a name="p44426301"></a><a name="p44426301"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.85%" headers="mcps1.1.4.1.3 "><p id="p111104535"><a name="p111104535"></a><a name="p111104535"></a>指定需要采集的Profiling数据内容范围，具体请参见<a href="ProfDataTypeConfig.md">ProfDataTypeConfig</a>。</p>
<p id="p41760635"><a name="p41760635"></a><a name="p41760635"></a>数值可按bit位或的方式组合表征，指定输出多类数据信息。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section65909368"></a>

<a name="table37231957"></a>
<table><thead align="left"><tr id="row10137733"><th class="cellrowborder" valign="top" width="14.06%" id="mcps1.1.4.1.1"><p id="p15850073"><a name="p15850073"></a><a name="p15850073"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.239999999999998%" id="mcps1.1.4.1.2"><p id="p8787537"><a name="p8787537"></a><a name="p8787537"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="72.7%" id="mcps1.1.4.1.3"><p id="p40701910"><a name="p40701910"></a><a name="p40701910"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row8520399"><td class="cellrowborder" valign="top" width="14.06%" headers="mcps1.1.4.1.1 "><p id="p19063717"><a name="p19063717"></a><a name="p19063717"></a>aclgrphProfConfig</p>
</td>
<td class="cellrowborder" valign="top" width="13.239999999999998%" headers="mcps1.1.4.1.2 "><p id="p657215"><a name="p657215"></a><a name="p657215"></a>aclgrphProfConfig</p>
</td>
<td class="cellrowborder" valign="top" width="72.7%" headers="mcps1.1.4.1.3 "><p id="p53234437"><a name="p53234437"></a><a name="p53234437"></a>Profiling配置信息结构体指针。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section56313405"></a>

-   aclgrphProfConfig类型数据可以只创建一次，多处使用，用户需要保证数据的一致性和准确性。
-   与[aclgrphProfDestroyConfig](aclgrphProfDestroyConfig.md)接口配对使用，先调用aclgrphProfCreateConfig接口再调用aclgrphProfDestroyConfig接口。
-   用户需保证程序结束时调用aclgrphProfDestroyConfig销毁所有创建的Profiling配置信息，否则可能会导致内存泄漏。
-   如果用户想在不同的Device上指定不同的Profiling配置信息，则可创建不同的aclgrphProfConfig类型数据，并依次调用[aclgrphProfStart](aclgrphProfStart.md)接口将不同的配置信息下发到不同的Device上。同时注意Device信息不能有重复。

