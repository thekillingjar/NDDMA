# aclgrphProfDestroyConfig<a name="ZH-CN_TOPIC_0000001312645889"></a>

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

## 功能说明<a name="section37522006"></a>

销毁Profiling配置信息。

## 函数原型<a name="section2153736"></a>

```
Status aclgrphProfDestroyConfig(aclgrphProfConfig *profiler_config)
```

## 参数说明<a name="section19383628"></a>

<a name="table15663995"></a>
<table><thead align="left"><tr id="row37441107"><th class="cellrowborder" valign="top" width="14.45%" id="mcps1.1.4.1.1"><p id="p12830815"><a name="p12830815"></a><a name="p12830815"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.690000000000001%" id="mcps1.1.4.1.2"><p id="p32663055"><a name="p32663055"></a><a name="p32663055"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.86%" id="mcps1.1.4.1.3"><p id="p19430358"><a name="p19430358"></a><a name="p19430358"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row23702140"><td class="cellrowborder" valign="top" width="14.45%" headers="mcps1.1.4.1.1 "><p id="p40825226"><a name="p40825226"></a><a name="p40825226"></a>profiler_config</p>
</td>
<td class="cellrowborder" valign="top" width="12.690000000000001%" headers="mcps1.1.4.1.2 "><p id="p18509020"><a name="p18509020"></a><a name="p18509020"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.86%" headers="mcps1.1.4.1.3 "><p id="p22835675"><a name="p22835675"></a><a name="p22835675"></a>Profiling配置信息结构指针。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section40234932"></a>

<a name="table4209123"></a>
<table><thead align="left"><tr id="row28073419"><th class="cellrowborder" valign="top" width="14.42%" id="mcps1.1.4.1.1"><p id="p59354473"><a name="p59354473"></a><a name="p59354473"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.19%" id="mcps1.1.4.1.2"><p id="p42983048"><a name="p42983048"></a><a name="p42983048"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="72.39%" id="mcps1.1.4.1.3"><p id="p59074896"><a name="p59074896"></a><a name="p59074896"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row20337256"><td class="cellrowborder" valign="top" width="14.42%" headers="mcps1.1.4.1.1 "><p id="p36705016"><a name="p36705016"></a><a name="p36705016"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.1.4.1.2 "><p id="p20316329"><a name="p20316329"></a><a name="p20316329"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="72.39%" headers="mcps1.1.4.1.3 "><p id="p35009954"><a name="p35009954"></a><a name="p35009954"></a>SUCCESS：成功。</p>
<p id="p46654138"><a name="p46654138"></a><a name="p46654138"></a>其他：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section26570072"></a>

-   与[aclgrphProfCreateConfig](aclgrphProfCreateConfig.md)接口配对使用，先调用aclgrphProfCreateConfig接口再调用aclgrphProfDestroyConfig接口。
-   需要销毁的Profiling配置信息必须是由aclgrphProfCreateConfig创建。
-   销毁Profiling配置信息失败，请检查配置信息参数是否合理。

