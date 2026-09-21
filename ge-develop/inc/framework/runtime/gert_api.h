/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_GERT_API_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_GERT_API_H_
#include "model_v2_executor.h"
#include "stream_executor.h"
#include "common/ge_types.h"
#include "memory/allocator_desc.h"
#include "exe_graph/lowering/lowering_opt.h"
#include "stream_allocator.h"
#include "event_allocator.h"
#include "rt_session.h"

namespace gert {
/**
 * Allocator 工厂类，创建Allocator
 */
class VISIBILITY_EXPORT AllocatorFactory {
 public:
  // 根据placement创建allocator
  static std::unique_ptr<ge::Allocator> Create(const std::string &graph_name, const TensorPlacement &placement);
  static std::unique_ptr<ge::Allocator> Create(const TensorPlacement &placement);
 private:
  AllocatorFactory() = default;
};

struct LoadExecutorArgs {
  RtSession *const rt_session;
  std::vector<ge::FileConstantMem> file_constant_mems;
};

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromFile(const ge::char_t *model_path, ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           ge::graphStatus &error_code);

/**
 * 将ModelData加载为Executor，  使用该接口创建executor时, 用户需要保证调用executor->Load传入相同的RtSession
 * @param model_data model_data从文件中读取后的内容
 * @param rt_session session对象，变量等资源会通过sesion共享
 * @param error_code 如果load失败（返回了空指针），那么本变量返回对应错误码
 * @return 成功时返回StreamExecutor的指针，失败时返回空指针。
 *         返回值类型是unique_ptr，因此返回的StreamExecutor生命周期由外部管理。
 */
VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelDataWithRtSession(const ge::ModelData &model_data,
                                                                        RtSession *const rt_session,
                                                                        ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           const LoadExecutorArgs &args,
                                                           ge::graphStatus &error_code);
VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           const ExecutorOption &executor_option,
                                                           ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<ModelV2Executor> LoadExecutorFromModelData(const ge::ModelData &model_data,
                                                           const ExecutorOption &executor_option,
                                                           StreamAllocator *const stream_allocator,
                                                           EventAllocator *const event_allocator,
                                                           NotifyAllocator *const notify_allocator,
                                                           ge::graphStatus &error_code);

VISIBILITY_EXPORT
std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data, const void *weight_ptr,
                                                                const size_t weight_size, ge::graphStatus &error_code);

/**
 * 将ModelData加载为StreamExecutor，本函数等同为LoadStreamExecutorFromModelData(model_data, {}, error_cde);
 * @param model_data model_data从文件中读取后的内容
 * @param error_code 如果load失败（返回了空指针），那么本变量返回对应错误码
 * @return 成功时返回StreamExecutor的指针，失败时返回空指针。
 *         返回值类型是unique_ptr，因此返回的StreamExecutor生命周期由外部管理。
 */
VISIBILITY_EXPORT
std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                ge::graphStatus &error_code);

/**
 * 将ModelData加载为StreamExecutor
 * @param model_data model_data从文件中读取后的内容
 * @param optimize_option 优化选项
 * @param error_code 如果load失败（返回了空指针），那么本变量返回对应错误码
 * @return 成功时返回StreamExecutor的指针，失败时返回空指针。
 *         返回值类型是unique_ptr，因此返回的StreamExecutor生命周期由外部管理。
 */

VISIBILITY_EXPORT
std::unique_ptr<StreamExecutor> LoadStreamExecutorFromModelData(const ge::ModelData &model_data,
                                                                const LoweringOption &optimize_option,
                                                                ge::graphStatus &error_code);
VISIBILITY_EXPORT
ge::graphStatus IsDynamicModel(const void *const model, size_t model_size, bool &is_dynamic_model);
VISIBILITY_EXPORT
ge::graphStatus IsDynamicModel(const ge::char_t *model_path, bool &is_dynamic_model);

VISIBILITY_EXPORT
ge::graphStatus LoadDataFromFile(const ge::char_t *model_path, ge::ModelData &model_data);

VISIBILITY_EXPORT
std::unique_ptr<ge::Allocator> CreateExternalAllocator(const ge::AllocatorDesc * const allocatorDesc);
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_GERT_API_H_
