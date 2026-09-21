# Stage<a name="ZH-CN_TOPIC_0000002516356795"></a>

## 产品支持情况<a name="section789110355111"></a>

<a name="zh-cn_topic_0000001312404881_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312404881_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001312404881_p1883113061818"><a name="zh-cn_topic_0000001312404881_p1883113061818"></a><a name="zh-cn_topic_0000001312404881_p1883113061818"></a><span id="zh-cn_topic_0000001312404881_ph20833205312295"><a name="zh-cn_topic_0000001312404881_ph20833205312295"></a><a name="zh-cn_topic_0000001312404881_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001312404881_p783113012187"><a name="zh-cn_topic_0000001312404881_p783113012187"></a><a name="zh-cn_topic_0000001312404881_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312404881_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312404881_p48327011813"><a name="zh-cn_topic_0000001312404881_p48327011813"></a><a name="zh-cn_topic_0000001312404881_p48327011813"></a><span id="zh-cn_topic_0000001312404881_ph583230201815"><a name="zh-cn_topic_0000001312404881_ph583230201815"></a><a name="zh-cn_topic_0000001312404881_ph583230201815"></a><term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312404881_p108715341013"><a name="zh-cn_topic_0000001312404881_p108715341013"></a><a name="zh-cn_topic_0000001312404881_p108715341013"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312404881_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312404881_p14832120181815"><a name="zh-cn_topic_0000001312404881_p14832120181815"></a><a name="zh-cn_topic_0000001312404881_p14832120181815"></a><span id="zh-cn_topic_0000001312404881_ph1483216010188"><a name="zh-cn_topic_0000001312404881_ph1483216010188"></a><a name="zh-cn_topic_0000001312404881_ph1483216010188"></a><term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312404881_zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312404881_p19948143911820"><a name="zh-cn_topic_0000001312404881_p19948143911820"></a><a name="zh-cn_topic_0000001312404881_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 头文件/库文件<a name="section710017525388"></a>

-   头文件：\#include <ge/fusion/pass/fusion\_pass\_reg.h\>
-   库文件：libge\_compiler.so

## 功能说明<a name="section15686020"></a>

融合Pass阶段定义。

注意：kAfterAssignLogicStream阶段不可注册普通融合Pass，此阶段不允许修改图结构，若误将Pass注册到此阶段，将被忽略。

## 函数原型<a name="section1610715016501"></a>

```
FusionPassRegistrationData &Stage(CustomPassStage stage)
```

## 参数说明<a name="section6956456"></a>

<a name="table33761356"></a>
<table><thead align="left"><tr id="row27598891"><th class="cellrowborder" valign="top" width="9.76%" id="mcps1.1.4.1.1"><p id="p20917673"><a name="p20917673"></a><a name="p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.93%" id="mcps1.1.4.1.2"><p id="p16609919"><a name="p16609919"></a><a name="p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.31%" id="mcps1.1.4.1.3"><p id="p59995477"><a name="p59995477"></a><a name="p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row27795493"><td class="cellrowborder" valign="top" width="9.76%" headers="mcps1.1.4.1.1 "><p id="p101143521351"><a name="p101143521351"></a><a name="p101143521351"></a>stage</p>
</td>
<td class="cellrowborder" valign="top" width="10.93%" headers="mcps1.1.4.1.2 "><p id="p171147522357"><a name="p171147522357"></a><a name="p171147522357"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.31%" headers="mcps1.1.4.1.3 "><p id="p1411455273512"><a name="p1411455273512"></a><a name="p1411455273512"></a>表示自定义融合Pass执行阶段。</p>
<a name="ul436915409308"></a><a name="ul436915409308"></a><ul id="ul436915409308"><li>kBeforeInferShape：（默认值）自定义Pass在框架入口处InferShape之前执行。</li><li>kAfterInferShape：自定义Pass在InferShape之后执行。<p id="p231024694316"><a name="p231024694316"></a><a name="p231024694316"></a>如果自定义Pass在InferShape之后执行，Pass中需要保证改图之后shape的连续性：</p>
<a name="screen13531144547"></a><a name="screen13531144547"></a><pre class="screen" codetype="Cpp" id="screen13531144547">    // 1. 获取输入节点node1的输出描述
    TensorDesc output_desc;
    node1.GetOutputDesc(0, output_desc);
    // 2. 更新当前节点node2的输入描述
    node2.UpdateInputDesc(0, output_desc);
    // 3. 当前节点node2推导InferShape
    operator2.InferShapeAndType();</pre>
</li><li>kAfterAssignLogicStream：自定义Pass在逻辑流分配阶段之后执行。该阶段仅接收逻辑流分配的Pass，由于该阶段不允许改图，其他场景的改图Pass会校验报错。</li><li>kAfterBuiltinFusionPass：自定义Pass在内置的原图融合Pass之后执行。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section62608109"></a>

返回FusionPassRegistrationData对象。

## 约束说明<a name="section1679583814442"></a>

无

