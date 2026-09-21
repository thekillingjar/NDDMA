# REGISTER\_CUSTOM\_PASS<a name="ZH-CN_TOPIC_0000002484436836"></a>

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

-   头文件：\#include <register/register\_custom\_pass.h\>
-   库文件：libregister.so

## 功能说明<a name="zh-cn_topic_0000001265084818_section212607105720"></a>

开发人员可以选择将改图函数注册到框架中，由框架在编译最开始调用自定义改图Pass，调用REGISTER\_CUSTOM\_PASS进行自定义Pass注册。

## 函数原型<a name="zh-cn_topic_0000001265084818_section129451113125413"></a>

```
REGISTER_CUSTOM_PASS(name) 
```

## 参数说明<a name="zh-cn_topic_0000001265084818_section1843519321017"></a>

<a name="zh-cn_topic_0000001265084818_table18350576"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265084818_row29692563"><th class="cellrowborder" valign="top" width="19.37%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265084818_p56287378"><a name="zh-cn_topic_0000001265084818_p56287378"></a><a name="zh-cn_topic_0000001265084818_p56287378"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.5%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265084818_p62983758"><a name="zh-cn_topic_0000001265084818_p62983758"></a><a name="zh-cn_topic_0000001265084818_p62983758"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="61.129999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265084818_p47160602"><a name="zh-cn_topic_0000001265084818_p47160602"></a><a name="zh-cn_topic_0000001265084818_p47160602"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265084818_row61912443"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265084818_p9757144353"><a name="zh-cn_topic_0000001265084818_p9757144353"></a><a name="zh-cn_topic_0000001265084818_p9757144353"></a>name</p>
</td>
<td class="cellrowborder" valign="top" width="19.5%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265084818_p13947197203216"><a name="zh-cn_topic_0000001265084818_p13947197203216"></a><a name="zh-cn_topic_0000001265084818_p13947197203216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.129999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265084818_p1476526254"><a name="zh-cn_topic_0000001265084818_p1476526254"></a><a name="zh-cn_topic_0000001265084818_p1476526254"></a>自定义Pass的名称。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001265084818_section8491941407"></a>

无

## 约束说明<a name="zh-cn_topic_0000001265084818_section65498832"></a>

调用时以REGISTER\_CUSTOM\_PASS开始，以“.”连接CustomPassFn等接口。例如：

```
#include "register/register_custom_pass.h"
REGISTER_CUSTOM_PASS("pass_name").CustomPassFn(CustomPassFunc);
```

