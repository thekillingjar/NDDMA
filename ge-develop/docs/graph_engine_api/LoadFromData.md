# LoadFromData<a name="ZH-CN_TOPIC_0000002516551429"></a>

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

## 头文件<a name="section6743111793111"></a>

\#include <graph/kernel\_launch\_info.h\>

## 功能说明<a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section36583473819"></a>

提供将序列化后的Task任务反序列化的能力，主要用于在GenerateTask函数中，将框架构造的Task反序列化后，修改其参数使用。

## 函数原型<a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section13230182415108"></a>

```
static KernelLaunchInfo LoadFromData(const gert::ExeResGenerationContext *context, const std::vector<uint8_t> &data)
```

## 参数说明<a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section75395119104"></a>

<a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p10223674448"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p10223674448"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p645511218169"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p645511218169"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p1922337124411"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p1922337124411"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p1247694711133"><a name="p1247694711133"></a><a name="p1247694711133"></a>context</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"></a><a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_p15663137127"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p182416224172"><a name="p182416224172"></a><a name="p182416224172"></a>GenrateTask函数的入参，保存了算子的基础信息。</p>
</td>
</tr>
<tr id="row1634125291313"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p434115221314"><a name="p434115221314"></a><a name="p434115221314"></a>data</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p18680152151411"><a name="p18680152151411"></a><a name="p18680152151411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p134114529136"><a name="p134114529136"></a><a name="p134114529136"></a>KernelLaunchInfo序列化后的结果。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section25791320141317"></a>

返回data入参反序列化后的结果。

## 约束说明<a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section320753512363"></a>

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
   ....
  // 更改原AI Core任务的ArgsFormat
  auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
  // 从task中获取argsformat
  auto aicore_args_format_str = aicore_task.GetArgsFormat();
  auto aicore_args_format = ArgsFormatSerializer::Deserialize(aicore_args_format_str);
  size_t i = 0UL;
  // 找到第一个输入地址的位置
  for (; i < aicore_args_format.size(); i++) {
    if (aicore_args_format[i].GetType() == ArgDescType::kIrInput ||
        aicore_args_format[i].GetType() == ArgDescType::kInputInstance) {
      break;
    }
  }
  // 在输入地址前插入一个通信地址
  aicore_args_format.insert(aicore_args_format.begin() + i, ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  // 将修改后的argsformat序列化后设置回Task中
  aicore_task.SetArgsFormat(ArgsFormatSerializer::Serialize(aicore_args_format).GetString());
  // 重新序列化并设置到Tasks中
  tasks.back() = aicore_task.Serialize();
  return SUCCESS;
}
```

