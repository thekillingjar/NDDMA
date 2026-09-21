# Serialize<a name="ZH-CN_TOPIC_0000002492435896"></a>

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

## 头文件<a name="section1762152904618"></a>

\#include <graph/arg\_desc\_info.h\>

## 函数功能<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section36893359"></a>

将ArgsFormat信息序列化成数据流。

## 函数原型<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section136951948195410"></a>

```
static AscendString Serialize(const std::vector<ArgDescInfo> &args_format)
```

## 参数说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section63604780"></a>

<a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p10223674448"><a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p10223674448"></a><a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p645511218169"><a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p645511218169"></a><a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p1922337124411"><a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p1922337124411"></a><a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p417191455213"><a name="p417191455213"></a><a name="p417191455213"></a>args_format</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p143401361158"><a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p143401361158"></a><a name="zh-cn_topic_0000001609859929_zh-cn_topic_0000001339510360_p143401361158"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p8234133625"><a name="p8234133625"></a><a name="p8234133625"></a>ArgsFormat信息，由若干个<a href="ArgDescInfo.md">ArgDescInfo</a>组成。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section35572112"></a>

返回ArgsFormat的序列化结果。

失败时返回空字符串。

## 约束说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section62768825"></a>

无

## 调用示例<a name="zh-cn_topic_0000001526046958_zh-cn_topic_0000001339347388_section320753512363"></a>

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks) {
....
// 更改原AI Core任务的ArgsFormat
  auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
  auto aicore_args_format_str = aicore_task.GetArgsFormat();
  auto aicore_args_format = ArgsFormatSerializer::Deserialize(aicore_args_format_str);
  size_t i = 0UL;
  for (; i < aicore_args_format.size(); i++) {
    if (aicore_args_format[i].GetType() == ArgDescType::kIrInput ||
        aicore_args_format[i].GetType() == ArgDescType::kInputInstance) {
      break;
    }
  }
  aicore_args_format.insert(aicore_args_format.begin() + i, ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  aicore_task.SetArgsFormat(ArgsFormatSerializer::Serialize(aicore_args_format).GetString());
  tasks.back() = aicore_task.Serialize();
  return SUCCESS;
}
```

