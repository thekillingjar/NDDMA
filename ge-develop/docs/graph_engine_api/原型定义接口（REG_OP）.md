# 原型定义接口（REG\_OP）<a name="ZH-CN_TOPIC_0000002516551457"></a>

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

## 头文件<a name="section995734502317"></a>

\#include <graph/operator\_reg.h\>

## 功能说明<a name="zh-cn_topic_0000001312403741_section13189358"></a>

定义算子的原型，包括算子的输入、输出、属性以及对应的数据类型。

进行如上算子原型定义后，即相当于向GE注册了该算子的原型，告知GE对应类型的算子应该具备哪些输入、输出与属性；同时相当于定义了一个op::xxx的Class，开发者可以include该原型头文件，然后实例化该Class进行IR模型构建，如下所示：

```
conv = op::Conv2D()
conv.set_input_x(feature_map_data)
conv.set_input_filter(weight_data)
```

具体的模型构建可以参考《图模式开发指南》。

## 函数原型<a name="zh-cn_topic_0000001312403741_section31291646"></a>

函数原型定义示例如下：

```
REG_OP(xxx)
    .INPUT(x1, type)
    .OPTIONAL_INPUT(x2, type)
    .DYNAMIC_INPUT(x3, type)
    .OUTPUT(y1, type)
    .DYNAMIC_OUTPUT(y3, type)
    .REQUIRED_ATTR(a, type)
    .ATTR(b, type, default_value)
    .GRAPH(z1)
    .DYNAMIC_GRAPH(z2)
    .OP_END_FACTORY_REG(xxx)
```

## 接口说明<a name="zh-cn_topic_0000001312403741_section51595365"></a>

<a name="zh-cn_topic_0000001312403741_table2051894852017"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312403741_row4558174815206"><th class="cellrowborder" valign="top" width="26.479999999999997%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312403741_p255884814201"><a name="zh-cn_topic_0000001312403741_p255884814201"></a><a name="zh-cn_topic_0000001312403741_p255884814201"></a><strong id="zh-cn_topic_0000001312403741_b145581148152018"><a name="zh-cn_topic_0000001312403741_b145581148152018"></a><a name="zh-cn_topic_0000001312403741_b145581148152018"></a>接口名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="43.45%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312403741_p14558184812200"><a name="zh-cn_topic_0000001312403741_p14558184812200"></a><a name="zh-cn_topic_0000001312403741_p14558184812200"></a><strong id="zh-cn_topic_0000001312403741_b955810482200"><a name="zh-cn_topic_0000001312403741_b955810482200"></a><a name="zh-cn_topic_0000001312403741_b955810482200"></a>接口说明</strong></p>
</th>
<th class="cellrowborder" valign="top" width="30.070000000000004%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312403741_p148796488197"><a name="zh-cn_topic_0000001312403741_p148796488197"></a><a name="zh-cn_topic_0000001312403741_p148796488197"></a>衍生接口（可用于IR模型构建）</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312403741_row35581048202018"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p3558184811208"><a name="zh-cn_topic_0000001312403741_p3558184811208"></a><a name="zh-cn_topic_0000001312403741_p3558184811208"></a>REG_OP(xxx)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p9558144816208"><a name="zh-cn_topic_0000001312403741_p9558144816208"></a><a name="zh-cn_topic_0000001312403741_p9558144816208"></a>定义一个算子原型，算子类型为xxx。</p>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p78791848161914"><a name="zh-cn_topic_0000001312403741_p78791848161914"></a><a name="zh-cn_topic_0000001312403741_p78791848161914"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section6311321202113">REG_OP</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row6558548162013"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p5558154852019"><a name="zh-cn_topic_0000001312403741_p5558154852019"></a><a name="zh-cn_topic_0000001312403741_p5558154852019"></a>.INPUT(x, type)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p054115053811"><a name="zh-cn_topic_0000001312403741_p054115053811"></a><a name="zh-cn_topic_0000001312403741_p054115053811"></a>定义输入名称（x）和类型(type)。</p>
<div class="p" id="zh-cn_topic_0000001312403741_p19549175963713"><a name="zh-cn_topic_0000001312403741_p19549175963713"></a><a name="zh-cn_topic_0000001312403741_p19549175963713"></a>类型为TensorType类型，例如：<a name="zh-cn_topic_0000001312403741_ul371442063918"></a><a name="zh-cn_topic_0000001312403741_ul371442063918"></a><ul id="zh-cn_topic_0000001312403741_ul371442063918"><li>TensorType{DT_FLOAT}</li><li>TensorType({DT_FLOAT, DT_INT8})</li><li>TensorType::ALL()</li></ul>
</div>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p15879164851912"><a name="zh-cn_topic_0000001312403741_p15879164851912"></a><a name="zh-cn_topic_0000001312403741_p15879164851912"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section191288192416">INPUT</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row9558748152018"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p555894832010"><a name="zh-cn_topic_0000001312403741_p555894832010"></a><a name="zh-cn_topic_0000001312403741_p555894832010"></a>.OPTIONAL_INPUT(x, type)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p1755814488207"><a name="zh-cn_topic_0000001312403741_p1755814488207"></a><a name="zh-cn_topic_0000001312403741_p1755814488207"></a>定义可选输入的名称（x）和类型（type）。</p>
<div class="p" id="zh-cn_topic_0000001312403741_p33527217403"><a name="zh-cn_topic_0000001312403741_p33527217403"></a><a name="zh-cn_topic_0000001312403741_p33527217403"></a>类型为TensorType类型，例如：<a name="zh-cn_topic_0000001312403741_ul93529210401"></a><a name="zh-cn_topic_0000001312403741_ul93529210401"></a><ul id="zh-cn_topic_0000001312403741_ul93529210401"><li>TensorType{DT_FLOAT}</li><li>TensorType({DT_FLOAT, DT_INT8})</li><li>TensorType::ALL()</li></ul>
</div>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p7879174831910"><a name="zh-cn_topic_0000001312403741_p7879174831910"></a><a name="zh-cn_topic_0000001312403741_p7879174831910"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section881812416243">OPTIONAL_INPUT</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row15558648142019"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p1558184802010"><a name="zh-cn_topic_0000001312403741_p1558184802010"></a><a name="zh-cn_topic_0000001312403741_p1558184802010"></a>.DYNAMIC_INPUT(x, type)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p85581848172016"><a name="zh-cn_topic_0000001312403741_p85581848172016"></a><a name="zh-cn_topic_0000001312403741_p85581848172016"></a>定义动态输入的名称（x）和类型（type）。</p>
<div class="p" id="zh-cn_topic_0000001312403741_p667172613403"><a name="zh-cn_topic_0000001312403741_p667172613403"></a><a name="zh-cn_topic_0000001312403741_p667172613403"></a>类型为TensorType类型，例如：<a name="zh-cn_topic_0000001312403741_ul26710262402"></a><a name="zh-cn_topic_0000001312403741_ul26710262402"></a><ul id="zh-cn_topic_0000001312403741_ul26710262402"><li>TensorType{DT_FLOAT}</li><li>TensorType({DT_FLOAT, DT_INT8})</li><li>TensorType::ALL()</li></ul>
</div>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p1987924801920"><a name="zh-cn_topic_0000001312403741_p1987924801920"></a><a name="zh-cn_topic_0000001312403741_p1987924801920"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section164801239255">DYNAMIC_INPUT</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row15558154872020"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p6558124810201"><a name="zh-cn_topic_0000001312403741_p6558124810201"></a><a name="zh-cn_topic_0000001312403741_p6558124810201"></a>.OUTPUT(x, type)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p4558124862019"><a name="zh-cn_topic_0000001312403741_p4558124862019"></a><a name="zh-cn_topic_0000001312403741_p4558124862019"></a>定义输出的名称（x）和类型（type）。</p>
<div class="p" id="zh-cn_topic_0000001312403741_p13982113917409"><a name="zh-cn_topic_0000001312403741_p13982113917409"></a><a name="zh-cn_topic_0000001312403741_p13982113917409"></a>类型为TensorType类型，例如：<a name="zh-cn_topic_0000001312403741_ul13982163914012"></a><a name="zh-cn_topic_0000001312403741_ul13982163914012"></a><ul id="zh-cn_topic_0000001312403741_ul13982163914012"><li>TensorType{DT_FLOAT}</li><li>TensorType({DT_FLOAT, DT_INT8})</li><li>TensorType::ALL()</li></ul>
</div>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p1387984817194"><a name="zh-cn_topic_0000001312403741_p1387984817194"></a><a name="zh-cn_topic_0000001312403741_p1387984817194"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section1537451932511">OUTPUT</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row1355816482206"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p555844817207"><a name="zh-cn_topic_0000001312403741_p555844817207"></a><a name="zh-cn_topic_0000001312403741_p555844817207"></a>.DYNAMIC_OUTPUT(x, type)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p6558648102015"><a name="zh-cn_topic_0000001312403741_p6558648102015"></a><a name="zh-cn_topic_0000001312403741_p6558648102015"></a>定义动态输出的名称（x）和类型（type）。</p>
<div class="p" id="zh-cn_topic_0000001312403741_p76934613412"><a name="zh-cn_topic_0000001312403741_p76934613412"></a><a name="zh-cn_topic_0000001312403741_p76934613412"></a>类型为TensorType类型，例如：<a name="zh-cn_topic_0000001312403741_ul4693769414"></a><a name="zh-cn_topic_0000001312403741_ul4693769414"></a><ul id="zh-cn_topic_0000001312403741_ul4693769414"><li>TensorType{DT_FLOAT}</li><li>TensorType({DT_FLOAT, DT_INT8})</li><li>TensorType::ALL()</li></ul>
</div>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p9879144861911"><a name="zh-cn_topic_0000001312403741_p9879144861911"></a><a name="zh-cn_topic_0000001312403741_p9879144861911"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section890024416252">DYNAMIC_OUTPUT</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row1055834842014"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p135581048202013"><a name="zh-cn_topic_0000001312403741_p135581048202013"></a><a name="zh-cn_topic_0000001312403741_p135581048202013"></a>.REQUIRED_ATTR(x, type)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p138601144162815"><a name="zh-cn_topic_0000001312403741_p138601144162815"></a><a name="zh-cn_topic_0000001312403741_p138601144162815"></a>定义必备属性的名称（x）和类型（type）。</p>
<div class="p" id="zh-cn_topic_0000001312403741_p115581648152013"><a name="zh-cn_topic_0000001312403741_p115581648152013"></a><a name="zh-cn_topic_0000001312403741_p115581648152013"></a>type的可选值包括：<a name="zh-cn_topic_0000001312403741_ul1086536114111"></a><a name="zh-cn_topic_0000001312403741_ul1086536114111"></a><ul id="zh-cn_topic_0000001312403741_ul1086536114111"><li>Int，属性类型为int64_t</li><li>Float，属性类型为float</li><li>String，属性类型为string</li><li>Bool，属性类型为bool</li><li>Tensor，属性类型为Tensor</li><li>Type，属性为Type枚举定义</li><li>NamedAttrs，属性类型为NamedAttrs</li><li>AscendString，属性类型为AscendString</li><li>ListInt，属性类型为vector&lt;int64_t&gt;，int64_t列表</li><li>ListFloat，属性类型为vector&lt;float&gt;，float列表</li><li>ListString，属性类型为vector&lt;string&gt;，string列表</li><li>ListBool，属性类型为vector&lt;bool&gt;，bool列表</li><li>ListTensor，属性类型为vector&lt;Tensor&gt;，Tensor列表</li><li>Bytes，属性类型为Buffer</li><li>ListType，属性类型为vector&lt;Type&gt;，Type列表</li><li>ListListInt，属性类型为vector&lt;vector&lt;int64_t&gt;&gt;，2维列表</li><li>ListAscendString，属性类型为vector&lt;AscendString&gt;，AscendString列表</li><li>ListNamedAttrs，属性类型为vector&lt;NamedAttrs&gt;，NamedAttrs列表</li></ul>
</div>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p15879134891912"><a name="zh-cn_topic_0000001312403741_p15879134891912"></a><a name="zh-cn_topic_0000001312403741_p15879134891912"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section14831019182612">REQUIRED_ATTR</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row9558548102013"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p11558848172018"><a name="zh-cn_topic_0000001312403741_p11558848172018"></a><a name="zh-cn_topic_0000001312403741_p11558848172018"></a>.ATTR(x, type, default_value)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p11864112224316"><a name="zh-cn_topic_0000001312403741_p11864112224316"></a><a name="zh-cn_topic_0000001312403741_p11864112224316"></a>定义可选属性的名称、类型以及默认值。</p>
<p id="zh-cn_topic_0000001312403741_p11984132216444"><a name="zh-cn_topic_0000001312403741_p11984132216444"></a><a name="zh-cn_topic_0000001312403741_p11984132216444"></a>当用户不设置算子对象的属性时，会使用此处设置的默认值。</p>
<div class="p" id="zh-cn_topic_0000001312403741_p7625648164616"><a name="zh-cn_topic_0000001312403741_p7625648164616"></a><a name="zh-cn_topic_0000001312403741_p7625648164616"></a>type的可选值包括：<a name="zh-cn_topic_0000002516551457_zh-cn_topic_0000001312403741_ul1086536114111"></a><a name="zh-cn_topic_0000002516551457_zh-cn_topic_0000001312403741_ul1086536114111"></a><ul id="zh-cn_topic_0000002516551457_zh-cn_topic_0000001312403741_ul1086536114111"><li>Int，属性类型为int64_t</li><li>Float，属性类型为float</li><li>String，属性类型为string</li><li>Bool，属性类型为bool</li><li>Tensor，属性类型为Tensor</li><li>Type，属性为Type枚举定义</li><li>NamedAttrs，属性类型为NamedAttrs</li><li>AscendString，属性类型为AscendString</li><li>ListInt，属性类型为vector&lt;int64_t&gt;，int64_t列表</li><li>ListFloat，属性类型为vector&lt;float&gt;，float列表</li><li>ListString，属性类型为vector&lt;string&gt;，string列表</li><li>ListBool，属性类型为vector&lt;bool&gt;，bool列表</li><li>ListTensor，属性类型为vector&lt;Tensor&gt;，Tensor列表</li><li>Bytes，属性类型为Buffer</li><li>ListType，属性类型为vector&lt;Type&gt;，Type列表</li><li>ListListInt，属性类型为vector&lt;vector&lt;int64_t&gt;&gt;，2维列表</li><li>ListAscendString，属性类型为vector&lt;AscendString&gt;，AscendString列表</li><li>ListNamedAttrs，属性类型为vector&lt;NamedAttrs&gt;，NamedAttrs列表</li></ul>
</div>
<p id="zh-cn_topic_0000001312403741_p1492534204612"><a name="zh-cn_topic_0000001312403741_p1492534204612"></a><a name="zh-cn_topic_0000001312403741_p1492534204612"></a>定义示例：</p>
<a name="zh-cn_topic_0000001312403741_ul1781683784615"></a><a name="zh-cn_topic_0000001312403741_ul1781683784615"></a><ul id="zh-cn_topic_0000001312403741_ul1781683784615"><li>.ATTR(mode, Int, 1)</li><li>.ATTR(pad, ListInt, {0, 0, 0, 0})</li></ul>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p38791248141915"><a name="zh-cn_topic_0000001312403741_p38791248141915"></a><a name="zh-cn_topic_0000001312403741_p38791248141915"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section6273123082620">ATTR</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row1753813597402"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p5538559204015"><a name="zh-cn_topic_0000001312403741_p5538559204015"></a><a name="zh-cn_topic_0000001312403741_p5538559204015"></a>.GRAPH(z1)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p255116911438"><a name="zh-cn_topic_0000001312403741_p255116911438"></a><a name="zh-cn_topic_0000001312403741_p255116911438"></a>注册算子中包含的子图信息，输入z1为子图名称。</p>
<p id="zh-cn_topic_0000001312403741_p826871910461"><a name="zh-cn_topic_0000001312403741_p826871910461"></a><a name="zh-cn_topic_0000001312403741_p826871910461"></a>例如If算子注册的子图为：</p>
<p id="zh-cn_topic_0000001312403741_p124772361465"><a name="zh-cn_topic_0000001312403741_p124772361465"></a><a name="zh-cn_topic_0000001312403741_p124772361465"></a>.GRAPH(then_branch)    .GRAPH(else_branch)</p>
<p id="zh-cn_topic_0000001312403741_p11539159154011"><a name="zh-cn_topic_0000001312403741_p11539159154011"></a><a name="zh-cn_topic_0000001312403741_p11539159154011"></a>对于同一个算子，注册的算子子图名称需要保持唯一。</p>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p1053905916403"><a name="zh-cn_topic_0000001312403741_p1053905916403"></a><a name="zh-cn_topic_0000001312403741_p1053905916403"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section4270146162918">GRAPH</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row2653562412"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p1165317620416"><a name="zh-cn_topic_0000001312403741_p1165317620416"></a><a name="zh-cn_topic_0000001312403741_p1165317620416"></a>.DYNAMIC_GRAPH(z2)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p186531615412"><a name="zh-cn_topic_0000001312403741_p186531615412"></a><a name="zh-cn_topic_0000001312403741_p186531615412"></a>注册动态算子子图信息，输入z2为子图名称。</p>
<p id="zh-cn_topic_0000001312403741_p53272479481"><a name="zh-cn_topic_0000001312403741_p53272479481"></a><a name="zh-cn_topic_0000001312403741_p53272479481"></a>例如Case算子注册的子图为：</p>
<p id="zh-cn_topic_0000001312403741_p739852016493"><a name="zh-cn_topic_0000001312403741_p739852016493"></a><a name="zh-cn_topic_0000001312403741_p739852016493"></a>.DYNAMIC_GRAPH(branches)</p>
<p id="zh-cn_topic_0000001312403741_p5556757529"><a name="zh-cn_topic_0000001312403741_p5556757529"></a><a name="zh-cn_topic_0000001312403741_p5556757529"></a>对于同一个算子，注册的算子子图名称需要保持唯一。</p>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p1465316114113"><a name="zh-cn_topic_0000001312403741_p1465316114113"></a><a name="zh-cn_topic_0000001312403741_p1465316114113"></a><a href="原型定义衍生接口.md#zh-cn_topic_0000001312484733_section5572531203010">DYNAMIC_GRAPH</a></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row1737252116277"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p123721321192714"><a name="zh-cn_topic_0000001312403741_p123721321192714"></a><a name="zh-cn_topic_0000001312403741_p123721321192714"></a>.INFER_SHAPE_AND_TYPE()</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p5372132122719"><a name="zh-cn_topic_0000001312403741_p5372132122719"></a><a name="zh-cn_topic_0000001312403741_p5372132122719"></a>该接口为历史遗留兼容性接口，当前版本用户无需使用。</p>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p153721721102720"><a name="zh-cn_topic_0000001312403741_p153721721102720"></a><a name="zh-cn_topic_0000001312403741_p153721721102720"></a>-</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312403741_row2558648152016"><td class="cellrowborder" valign="top" width="26.479999999999997%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312403741_p5559144892016"><a name="zh-cn_topic_0000001312403741_p5559144892016"></a><a name="zh-cn_topic_0000001312403741_p5559144892016"></a>.OP_END_FACTORY_REG(x)</p>
</td>
<td class="cellrowborder" valign="top" width="43.45%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312403741_p955974822010"><a name="zh-cn_topic_0000001312403741_p955974822010"></a><a name="zh-cn_topic_0000001312403741_p955974822010"></a>与REG_OP配对，结束算子原型定义。</p>
<p id="zh-cn_topic_0000001312403741_p20944114615019"><a name="zh-cn_topic_0000001312403741_p20944114615019"></a><a name="zh-cn_topic_0000001312403741_p20944114615019"></a>算子类型（x）与REG_OP(x)中的类型相同。</p>
</td>
<td class="cellrowborder" valign="top" width="30.070000000000004%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312403741_p178791648151913"><a name="zh-cn_topic_0000001312403741_p178791648151913"></a><a name="zh-cn_topic_0000001312403741_p178791648151913"></a>-</p>
</td>
</tr>
</tbody>
</table>

>![](public_sys-resources/icon-note.gif) **说明：** 
>OpReg类中的OpReg &N\(\)接口的功能是为了用户进行算子注册的时候，使用.\*\*的方式调用OpReg类的接口，例如.INPUT\(x, type\)、.OUTPUT\(x, type\)，无其他含义。

## 返回值说明<a name="zh-cn_topic_0000001312403741_section61705107"></a>

无

## 约束说明<a name="zh-cn_topic_0000001312403741_section18475057"></a>

-   REG\_OP的算子类型必须全局唯一。
-   同一个算子的输入名称之间不能重复。
-   同一个算子的输出名称之间不能重复。
-   同一个算子的属性名称之间不能重复。

## 调用示例和相关API<a name="zh-cn_topic_0000001312403741_section46544415"></a>

动态输入的算子原型定义示例：

```
REG_OP(AddN)
    .DYNAMIC_INPUT(x, TensorType({NumberType(), DT_VARIANT}))
    .OUTPUT(y, TensorType({NumberType(), DT_VARIANT}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(AddN)
```

多输入的算子原型定义示例：

```
REG_OP(GreaterEqual)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(GreaterEqual)
```

注册子图的算子原型定义示例：

```
REG_OP(If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(If)
```

