# ArgDescInfo构造函数和析构函数<a name="ZH-CN_TOPIC_0000002484311486"></a>

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

## 功能说明<a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_section36583473819"></a>

ArgDescInfo构造函数和析构函数。

## 函数原型<a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_section13230182415108"></a>

```
explicit ArgDescInfo(ArgDescType arg_type, int32_t ir_index = -1, bool is_folded = false)
~ArgDescInfo()
ArgDescInfo(const ArgDescInfo &other)
ArgDescInfo(ArgDescInfo &&other) noexcept
ArgDescInfo &operator=(const ArgDescInfo &other)
ArgDescInfo &operator=(ArgDescInfo &&other) noexcept
```

## 参数说明<a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_section75395119104"></a>

<a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p10223674448"><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p10223674448"></a><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p645511218169"><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p645511218169"></a><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p1922337124411"><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p1922337124411"></a><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p2340183613156"><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p2340183613156"></a><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p2340183613156"></a>arg_type</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p143401361158"><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p143401361158"></a><a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_p143401361158"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p1747263311424"><a name="p1747263311424"></a><a name="p1747263311424"></a>当前Args地址的类型。具体类型定义请参考<a href="ArgDescType.md">ArgDescType</a>。</p>
</td>
</tr>
<tr id="row19210241124219"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p192111741104219"><a name="p192111741104219"></a><a name="p192111741104219"></a>ir_index</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p197338485423"><a name="p197338485423"></a><a name="p197338485423"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p1021112415422"><a name="p1021112415422"></a><a name="p1021112415422"></a>当前Args地址对应的算子IR索引。</p>
</td>
</tr>
<tr id="row19291643104218"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p1729214318426"><a name="p1729214318426"></a><a name="p1729214318426"></a>is_folded</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="p10563164924218"><a name="p10563164924218"></a><a name="p10563164924218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p15292164316421"><a name="p15292164316421"></a><a name="p15292164316421"></a>当前地址是否需要被折叠成二级指针。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1213523864614"></a>

无

## 约束说明<a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001576727157_zh-cn_topic_0000001389787301_section320753512363"></a>

```
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks) {
...
  // 设置AI CPU任务
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context, "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  std::vector<ArgDescInfo> aicpu_args_format;
  // 构造了一个类型为kIrOutputDesc，ir_index为0，需要被折叠成二级指针的地址描述信息
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrOutputDesc, 0, true));
...
}
```

