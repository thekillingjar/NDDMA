/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_GRAPH_KERNEL_LAUNCH_INFO_H
#define METADEF_INC_EXTERNAL_GRAPH_KERNEL_LAUNCH_INFO_H

#include <vector>
#include <cstdint>
#include <memory>
#include "exe_graph/runtime/exe_res_generation_context.h"

namespace ge {
class KernelLaunchInfoImpl;
using KernelLaunchInfoImplPtr = std::unique_ptr<KernelLaunchInfoImpl>;
class KernelLaunchInfo {
 public:
  ~KernelLaunchInfo();
  KernelLaunchInfo(const KernelLaunchInfo &other);
  KernelLaunchInfo(KernelLaunchInfo &&other) noexcept;
  KernelLaunchInfo &operator=(const KernelLaunchInfo &other);
  KernelLaunchInfo &operator=(KernelLaunchInfo &&other) noexcept;

  /**
   * 从字符串中加载算子的Launch信息
   * @param context gentask callback函数的入参，保存了算子的基础信息
   * @param data 算子launch信息的序列化数据流
   * @return KernelLaunchInfo对象，保存了算子的Launch信息
   */
  static KernelLaunchInfo LoadFromData(const gert::ExeResGenerationContext *context,
      const std::vector<uint8_t> &data);
  /**
   * 创建一个Aicpu通信算子Task
   * @param context gentask callback函数的入参，保存了算子的基础信息
   * @param so_name aicpu算子的so名字
   * @param kernel_name aicpu算子的入口函数名字
   * @return KernelLaunchInfo对象，保存了算子的Launch信息
   */
  static KernelLaunchInfo CreateAicpuKfcTask(const gert::ExeResGenerationContext *context,
      const char *so_name, const char *kernel_name);
  /**
   * 创建一个Record Task，用于唤醒相同group_name的Wait Task
   * @param context gentask callback函数的入参，保存了算子的基础信息
   * @param group_name Record task的分组名字，默认为group
   * @return KernelLaunchInfo对象，保存了算子的Launch信息
   */
  static KernelLaunchInfo CreateHcomRecordTask(const gert::ExeResGenerationContext *context,
      const char *group_name = "group");
  /**
   * 创建一个Wait Task，用于阻塞当前流，当有相同group_name的Record Task被执行时，解除阻塞
   * @param context gentask callback函数的入参，保存了算子的基础信息
   * @param group_name Wait task的分组名字，默认为group
   * @return KernelLaunchInfo对象，保存了算子的Launch信息
   */
  static KernelLaunchInfo CreateHcomWaitTask(const gert::ExeResGenerationContext *context,
      const char *group_name = "group");
  /**
   * 创建一个FusionTask
   * @param context gentask callback函数的入参，保存了算子的基础信息
   * @param sub_tasks KernelLaunchInfo序列，包含aicore和ccu类型的KernelLaunchInfo
   * @return KernelLaunchInfo对象，保存了算子的Launch信息
   */
  static KernelLaunchInfo CreateFusionTask(const gert::ExeResGenerationContext *context,
      const std::vector<KernelLaunchInfo>& sub_tasks);
  /**
   * 创建一个Ccu Task
   * @param context gentask callback函数的入参，保存了算子的基础信息
   * @param groups CcuSubFusionTask 保存到group信息
   * @return KernelLaunchInfo对象，保存了算子的Launch信息
   */
  static KernelLaunchInfo CreateCcuTask(const gert::ExeResGenerationContext *context,
      const std::vector<std::string>& groups);
  /**
   * 将KernelLaunchInfo序列化成数据流
   * @return 被序列化后的数据流
   */
  std::vector<uint8_t> Serialize();
  /**
   * 获取当前task所在流的id
   * @return 当KernelLaucnhInfo合法时，返回当前task所在流的id(默认值为0)，非法时返回int32_max
   */
  uint32_t GetStreamId() const;
  /**
   * 设置task的流id
   * @param stream_id 流id
   */
  void SetStreamId(uint32_t stream_id);
  /**
   * 获取算子blockdim
   * @return 当KernelLaucnhInfo合法时，返回当前算子的blockdim(默认值为0)，非法时返回int32_max
   */
  uint32_t GetBlockDim() const;
  /**
   * 设置blockdim
   * @param block_dim 算子blockdim
   * @return SUCCESS: 设置成功，其他：设置失败报错
   */
  graphStatus SetBlockDim(uint32_t block_dim);
  /**
   * 获取当前task的args_format, args_format信息是args内存的语义化表达，用户通过拼接一个argsFormat内容，告诉框架如何排布args内存，
   * 只有aicpu和aicore算子有argsformat信息
   * @return 算子的args_format被设置时，返回args_format的序列化字符串，未设置时返回nullptr
   */
  const char *GetArgsFormat() const;
  /**
   * 设置当前task的args_format, args_format信息是args内存的语义化表达，用户通过拼接一个argsFormat内容，告诉框架如何排布args内存，
   * 只有aicpu和aicore算子有argsformat信息
   * @param args_format 算子的args_format信息
   * @return SUCCESS: 设置成功，其他：设置失败报错
   */
  graphStatus SetArgsFormat(const char *args_format);
  /**
   * 获取当前task的so_name, 只有aicpu算子可以获取到
   * @return 算子的so_name被设置时，返回so_name的字符串，未设置时返回nullptr
   */
  const char *GetSoName() const;
  /**
   * 获取当前task的kernel_name, 只有aicpu算子可以获取到
   * @return 算子的kernel_name被设置时，返回kernel_name的字符串，未设置时返回nullptr
   */
  const char *GetKernelName() const;
 private:
  KernelLaunchInfo() = delete;
  explicit KernelLaunchInfo(KernelLaunchInfoImplPtr &&impl);
  std::unique_ptr<KernelLaunchInfoImpl> impl_;
};
}
#endif // METADEF_INC_EXTERNAL_GRAPH_KERNEL_LAUNCH_INFO_H