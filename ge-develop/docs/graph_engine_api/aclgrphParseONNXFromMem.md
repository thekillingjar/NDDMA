# aclgrphParseONNXFromMem<a name="ZH-CN_TOPIC_0000001312645861"></a>

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

-   头文件：\#include <parser/onnx\_parser.h\>

    库文件：libfmk\_onnx\_parser.so

## 功能说明<a name="zh-cn_topic_0235751031_section15686020"></a>

将加载至内存的ONNX模型解析为Graph。

## 函数原型<a name="zh-cn_topic_0235751031_section1610715016501"></a>

```
graphStatus aclgrphParseONNXFromMem(const char *buffer, size_t size, const std::map<ge::AscendString, ge::AscendString> &parser_params, ge::Graph &graph)
```

## 参数说明<a name="section1128422614272"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="12.43%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.06%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="76.51%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0235751031_row27795493"><td class="cellrowborder" valign="top" width="12.43%" headers="mcps1.1.4.1.1 "><p id="p2015210314211"><a name="p2015210314211"></a><a name="p2015210314211"></a>buffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0235751031_p31450711"><a name="zh-cn_topic_0235751031_p31450711"></a><a name="zh-cn_topic_0235751031_p31450711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.51%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0235751031_p737611715492"><a name="zh-cn_topic_0235751031_p737611715492"></a><a name="zh-cn_topic_0235751031_p737611715492"></a>ONNX模型以二进制load到内存缓冲区的指针，尚未反序列化。</p>
</td>
</tr>
<tr id="row3982133893115"><td class="cellrowborder" valign="top" width="12.43%" headers="mcps1.1.4.1.1 "><p id="p1198373883118"><a name="p1198373883118"></a><a name="p1198373883118"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.2 "><p id="p698319383319"><a name="p698319383319"></a><a name="p698319383319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.51%" headers="mcps1.1.4.1.3 "><p id="p189831438113116"><a name="p189831438113116"></a><a name="p189831438113116"></a>内存缓冲区的大小。</p>
</td>
</tr>
<tr id="row57841253111110"><td class="cellrowborder" valign="top" width="12.43%" headers="mcps1.1.4.1.1 "><p id="p147844532118"><a name="p147844532118"></a><a name="p147844532118"></a>parser_params</p>
</td>
<td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.2 "><p id="p27847535117"><a name="p27847535117"></a><a name="p27847535117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.51%" headers="mcps1.1.4.1.3 "><p id="p83431614162713"><a name="p83431614162713"></a><a name="p83431614162713"></a>配置参数map映射表，key为参数类型，value为参数值，均为AscendString格式，用于描述原始模型解析参数。</p>
<p id="p737611715492"><a name="p737611715492"></a><a name="p737611715492"></a>map中支持的配置参数请参见<a href="Parser解析接口支持的配置参数.md">Parser解析接口支持的配置参数</a>。</p>
</td>
</tr>
<tr id="row15846111914516"><td class="cellrowborder" valign="top" width="12.43%" headers="mcps1.1.4.1.1 "><p id="p384671934511"><a name="p384671934511"></a><a name="p384671934511"></a>graph</p>
</td>
<td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.1.4.1.2 "><p id="p4846161916455"><a name="p4846161916455"></a><a name="p4846161916455"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="76.51%" headers="mcps1.1.4.1.3 "><p id="p1084617197452"><a name="p1084617197452"></a><a name="p1084617197452"></a>解析后生成的Graph。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0235751031_section62608109"></a>

<a name="zh-cn_topic_0235751031_table2601186"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row1832323"><th class="cellrowborder" valign="top" width="9.84%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p14200498"><a name="zh-cn_topic_0235751031_p14200498"></a><a name="zh-cn_topic_0235751031_p14200498"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.280000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p9389685"><a name="zh-cn_topic_0235751031_p9389685"></a><a name="zh-cn_topic_0235751031_p9389685"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="79.88%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p22367029"><a name="zh-cn_topic_0235751031_p22367029"></a><a name="zh-cn_topic_0235751031_p22367029"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0235751031_row66898905"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0235751031_p50102218"><a name="zh-cn_topic_0235751031_p50102218"></a><a name="zh-cn_topic_0235751031_p50102218"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="10.280000000000001%" headers="mcps1.1.4.1.2 "><p id="p158855407451"><a name="p158855407451"></a><a name="p158855407451"></a>graphStatus</p>
</td>
<td class="cellrowborder" valign="top" width="79.88%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0235751031_p918814449444"><a name="zh-cn_topic_0235751031_p918814449444"></a><a name="zh-cn_topic_0235751031_p918814449444"></a>GRAPH_SUCCESS(0)：成功。</p>
<p id="zh-cn_topic_0235751031_p91883446443"><a name="zh-cn_topic_0235751031_p91883446443"></a><a name="zh-cn_topic_0235751031_p91883446443"></a>其他值：失败。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0235751031_section38092103"></a>

无

