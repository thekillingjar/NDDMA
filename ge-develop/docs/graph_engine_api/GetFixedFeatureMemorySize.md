# GetFixedFeatureMemorySize<a name="ZH-CN_TOPIC_0000002516573393"></a>

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

## 头文件/库文件<a name="section3132657143710"></a>

-   头文件：\#include <ge/ge\_graph\_compile\_summary.h\>
-   库文件：libge\_compiler.so

## 功能说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section36893359"></a>

获取编译后固定Feature内存总大小（适用于静态shape图和动态shape图）。

## 函数原型<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section136951948195410"></a>

```
Status GetFixedFeatureMemorySize(size_t &size) const
```

## 参数说明<a name="section144401754174018"></a>

<a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="11.16%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.24%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.60000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0204328224_zh-cn_topic_0182636394_p66531170"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0204328224_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p1963855292513"><a name="p1963855292513"></a><a name="p1963855292513"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.1.4.1.2 "><p id="p176382521257"><a name="p176382521257"></a><a name="p176382521257"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="79.60000000000001%" headers="mcps1.1.4.1.3 "><p id="p1263885218251"><a name="p1263885218251"></a><a name="p1263885218251"></a>常量内存大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section35572112"></a>

<a name="table4209123"></a>
<table><thead align="left"><tr id="row28073419"><th class="cellrowborder" valign="top" width="11.1%" id="mcps1.1.4.1.1"><p id="p59354473"><a name="p59354473"></a><a name="p59354473"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.959999999999999%" id="mcps1.1.4.1.2"><p id="p42983048"><a name="p42983048"></a><a name="p42983048"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="78.94%" id="mcps1.1.4.1.3"><p id="p59074896"><a name="p59074896"></a><a name="p59074896"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row20337256"><td class="cellrowborder" valign="top" width="11.1%" headers="mcps1.1.4.1.1 "><p id="p36705016"><a name="p36705016"></a><a name="p36705016"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="9.959999999999999%" headers="mcps1.1.4.1.2 "><p id="p20316329"><a name="p20316329"></a><a name="p20316329"></a>Status</p>
</td>
<td class="cellrowborder" valign="top" width="78.94%" headers="mcps1.1.4.1.3 "><a name="ul1343942615010"></a><a name="ul1343942615010"></a><ul id="ul1343942615010"><li>SUCCESS：接口调用成功。</li><li>FAILED：接口调用失败。</li></ul>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section62768825"></a>

无

