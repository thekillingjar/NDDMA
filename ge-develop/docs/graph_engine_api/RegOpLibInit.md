# RegOpLibInit<a name="ZH-CN_TOPIC_0000002516551445"></a>

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

## 头文件<a name="section394375712335"></a>

\#include <register/op\_lib\_register.h\>

## 功能说明<a name="zh-cn_topic_0000001936560960_section36583473819"></a>

注册自定义算子动态库的初始化函数。

## 函数原型<a name="zh-cn_topic_0000001936560960_section13230182415108"></a>

```
OpLibRegister &RegOpLibInit(OpLibInitFunc func)
```

## 参数说明<a name="zh-cn_topic_0000001936560960_section75395119104"></a>

<a name="zh-cn_topic_0000001936560960_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001936560960_row6223476444"><th class="cellrowborder" valign="top" width="14.17%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001936560960_p10223674448"><a name="zh-cn_topic_0000001936560960_p10223674448"></a><a name="zh-cn_topic_0000001936560960_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.68%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001936560960_p645511218169"><a name="zh-cn_topic_0000001936560960_p645511218169"></a><a name="zh-cn_topic_0000001936560960_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.15%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001936560960_p1922337124411"><a name="zh-cn_topic_0000001936560960_p1922337124411"></a><a name="zh-cn_topic_0000001936560960_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001936560960_row152234713443"><td class="cellrowborder" valign="top" width="14.17%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001936560960_p1374621162312"><a name="zh-cn_topic_0000001936560960_p1374621162312"></a><a name="zh-cn_topic_0000001936560960_p1374621162312"></a>func</p>
</td>
<td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001936560960_p107457119238"><a name="zh-cn_topic_0000001936560960_p107457119238"></a><a name="zh-cn_topic_0000001936560960_p107457119238"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.15%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001936560960_p167457119233"><a name="zh-cn_topic_0000001936560960_p167457119233"></a><a name="zh-cn_topic_0000001936560960_p167457119233"></a>要注册的自定义初始化函数，类型为OpLibInitFunc。</p>
<a name="screen9923814261"></a><a name="screen9923814261"></a><pre class="screen" codetype="Cpp" id="screen9923814261">using OpLibInitFunc = uint32_t (*)(ge::AscendString&amp;);</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001936560960_section25791320141317"></a>

返回一个OpLibRegister对象，该对象新增注册了OpLibInitFunc函数func。

## 约束说明<a name="zh-cn_topic_0000001936560960_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001936560960_section320753512363"></a>

```
uint32_t Init(ge::AscendString&) {
  // init func
  return 0;
}

REGISTER_OP_LIB(vendor_1).RegOpLibInit(Init); // 注册厂商名为vendor_1的初始化函数Init
```

