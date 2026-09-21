# SetCustomValue<a name="ZH-CN_TOPIC_0000002516431415"></a>

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

## 功能说明<a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_section36583473819"></a>

设置ArgDescInfo的自定义值，只有type为kCustomValue时，才能设置成功。

## 函数原型<a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_section13230182415108"></a>

```
graphStatus SetCustomValue(uint64_t custom_value)
```

## 参数说明<a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_section75395119104"></a>

<a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p10223674448"><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p10223674448"></a><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p645511218169"><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p645511218169"></a><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p1922337124411"><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p1922337124411"></a><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p2340183613156"><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p2340183613156"></a><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p2340183613156"></a>custom_value</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p88767238815"><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p88767238815"></a><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p88767238815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p131917964515"><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p131917964515"></a><a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_p131917964515"></a>自定义值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_section25791320141317"></a>

设置成功时返回“ge::GRAPH\_SUCCESS”。

## 约束说明<a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000002027834817_zh-cn_topic_0000001339507276_section320753512363"></a>

```
// 需要存储在Args中的结构体
struct HcclCommParamDesc {
  uint64_t version : 4;
  uint64_t group_num : 4;
  uint64_t has_ffts : 1;
  uint64_t tiling_off : 7;
  uint64_t is_dyn : 48;
};

graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
...
  // 设置AI CPU任务
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
      "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  size_t input_size = context->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = context->GetComputeNodeInfo()->GetIrOutputsNum();
  const size_t offset = 3UL;
  union {
    HcclCommParamDesc hccl_desc;
    uint64_t custom_value;
  } desc;
  // 赋值
  desc.hccl_desc.version = 1;
  desc.hccl_desc.group_num = 1;
  desc.hccl_desc.has_ffts = 0;
  desc.hccl_desc.tiling_off = offset + input_size + output_size;
  desc.hccl_desc.is_dyn = 0;
  std::vector<ArgDescInfo> aicpu_args_format;
  ArgDescInfo custom_value_arg(ArgDescType::kCustomValue);
  // 设置自定义值
  custom_value_arg.SetCustomValue(desc.custom_value);
  // 将此结构体的内容转化成uint64_t的数字保存到ArgsFormat中
  aicpu_args_format.emplace_back(custom_value_arg)
...
}
```

