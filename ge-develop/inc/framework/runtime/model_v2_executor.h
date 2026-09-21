/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
#define AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
#include <memory>
#include <cstdlib>
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/ge_error_codes.h"
#include "model_desc.h"
#include "runtime/stream.h"
#include "exe_graph/runtime/tensor.h"
#include "common/ge_visibility.h"
#include "exe_graph_resource_guard.h"
#include "exe_graph_executor.h"
#include "subscriber/executor_subscribers_scheduler.h"
#include "common/ge_types.h"
#include "mem_allocator.h"
#include "framework/runtime/rt_session.h"
#include "framework/common/ge_types.h"
#include "framework/runtime/executor_option/executor_option.h"
#include "framework/runtime/stream_allocator.h"
#include "framework/runtime/event_allocator.h"
#include "common/host_resource_center/host_resource_center.h"
namespace gert {
enum class ExecutorState { kInit, kLoaded };
inline const ge::char_t *GetSubExeGraphTypeStr(const SubExeGraphType type) {
  constexpr const ge::char_t *kSubExeGraphTypeStrs[kSubExeGraphTypeEnd] = {"Init", "Main", "DeInit"};
  return kSubExeGraphTypeStrs[type];
}

enum class ExecuteArgIndex {
  kNum = 4,
  kNotifies = -1 * kNum,
  kRtEvents,
  kExternalAllocator,
  kStream,
  kEnd,
};

struct OuterWeightMem {
  const void *weight_ptr;  // this mem must be avalable and used for the same model
  size_t weight_size;
};

struct ModelExecuteArg {
  /**
   * 如果外部传入的stream不为空，那么本模型将执行在外部传入的流上。
   */
  rtStream_t stream;

  /**
   * 是否使用外部的allocator来申请流上内存。如果本成员指针为空，那么执行器会自行创建一个默认allocator使用。
   * 由于Host与Device之间下发任务后，Device总是在流上异步执行这个任务，因此allocator需要满足如下几个要求：
   * 1. 一个allocator仅对应唯一的stream
   * 2. 在对应的流同步之前，allocator内存池中的内存不可以归还到操作系统
   * 3. 在对应的流同步之前，allocator不可以被析构（原理同2）
   */
  Allocators *external_allocator;

  /**
   * 是否使用外部的stream allocator来申请辅流。如果本成员指针为空，那么执行器会自行创建一个默认的stream allocator使用。
   * 该stream allocator与ModelExecuteArg的第一个参数stream（外部主流）生命周期一致。若切换主流，也需要对应切换该allocator
   */
  StreamAllocator *external_stream_allocator;
  /**
   * 是否使用外部的event allocator来申请event。如果本成员指针为空，那么执行器会自行创建一个默认的event allocator使用。
   * 该event allocator与ModelExecuteArg的第一个参数stream（外部主流）生命周期一致。若切换主流，也需要对应切换该allocator
   * event allocator与stream allocator需要同时传入
   */
  EventAllocator *external_event_allocator;
  /**
   * 是否使用外部的notify allocator来申请notify。如果本成员指针为空，那么执行器会自行创建一个默认的notify
   * allocator使用。 该notify
   * allocator与ModelExecuteArg的第一个参数stream（外部主流）生命周期一致。若切换主流，也需要对应切换该allocator notify
   * allocator与stream allocator需要同时传入
   */
  NotifyAllocator *external_notify_allocator;

  uint64_t reserved[8];

  ModelExecuteArg() : ModelExecuteArg(nullptr, nullptr) {}
  ModelExecuteArg(const rtStream_t stream_, Allocators *const external_allocator_ = nullptr)
      : stream(stream_),
        external_allocator(external_allocator_),
        external_stream_allocator{nullptr},
        external_event_allocator{nullptr},
        external_notify_allocator{nullptr},
        reserved{0U} {}
};
static_assert(std::is_standard_layout<ModelExecuteArg>::value, "The class ModelExecuteArg must be a POD");

struct ModelLoadArg {
  RtSession *rt_session;
  OuterWeightMem outer_weight_mem;
  ModelLoadArg() : rt_session(nullptr), outer_weight_mem({nullptr, 0U}) {}
  ModelLoadArg(RtSession *rt_session_tmp = nullptr, OuterWeightMem outer_weight_mem_tmp = {nullptr, 0U})
      : rt_session(rt_session_tmp),
        outer_weight_mem(outer_weight_mem_tmp) {}
};

static_assert(std::is_standard_layout<ModelLoadArg>::value, "The class ModelLoadArg must be a POD");

class VISIBILITY_EXPORT ModelV2Executor {
 public:
  static std::unique_ptr<ModelV2Executor> Create(const ge::ExecuteGraphPtr &exe_graph, const ge::ModelData &model_data,
                                                 const std::shared_ptr<ge::GeRootModel> &root_model,
                                                 RtSession *session = nullptr);
  static std::unique_ptr<ModelV2Executor> Create(const ge::ExecuteGraphPtr &exe_graph,
                                                 const std::shared_ptr<ge::GeRootModel> &root_model,
                                                 RtSession *session = nullptr);
  static std::unique_ptr<ModelV2Executor> Create(const ge::ExecuteGraphPtr &exe_graph, const ExecutorOption &option,
                                                 const std::shared_ptr<ge::GeRootModel> &root_model,
                                                 RtSession *session = nullptr);

  ge::graphStatus Load();
  /**
   * 加载模型，本接口需要在模型执行前被调用。加载流程会完成模型的初始化、将重要数据拷贝到NPU等整个模型生命周期内仅需要执行一次的行为。
   * @param arg
   * 模型的执行参数，需要注意的是，此处传入的执行参数应该与Execute接口传入的执行参数具有相同的stream和allocator，
   *            否则在load完成后，外部需要调用流同步以保证不出现时序问题
   * @return 成功时返回`ge::GRAPH_SUCCESS`
   */
  ge::graphStatus Load(const ModelExecuteArg &arg);

  /**
   * 加载模型，本接口需要在模型执行前被调用。加载流程会完成模型的初始化、将重要数据拷贝到NPU等整个模型生命周期内仅需要执行一次的行为。
   * @param execute_arg
   * 模型的执行参数，需要注意的是，此处传入的执行参数应该与Execute接口传入的执行参数具有相同的stream和allocator，
   *            否则在load完成后，外部需要调用流同步以保证不出现时序问题
   * @param load_arg 模型的加载参数，加载时由外部传入, 执行时不会改变的参数,如RtSession.
   * @return 成功时返回`ge::GRAPH_SUCCESS`
   */
  ge::graphStatus Load(const ModelExecuteArg &arg, const ModelLoadArg &load_arg);

  /**
   * 异步执行模型，本接口将模型异步下发到NPU执行，本接口返回不代表模型执行完成，用户需要手动调用流同步等待模型执行完成。
   * 调用本接口前，请确保已经调用`Load`接口
   *
   * 用户可以通过多种方式指定输出Tensor，其行为分别为：
   *
   * * 调用本接口前，用户自行申请了足量空间的输出内存，并通过输出Tensor传入：执行完成后，输出内容被写入到用户申请的输出Tensor。
   *   若用户申请的输出Tensor不够长，那么本接口返回失败。
   * * 用户生成了输出Tensor，但是没有申请输出内存，将不包含输出内存的Tensor传入：本接口内部主动申请输出内存，并将输出内存传出。
   *   若用户没有在arg中指定Allocator，那么本接口输出的内存生命周期与本Executor一致；
   *   如果用户在arg中传入了Allocator，那么输出内存将使用用户传入的Allocator申请
   *
   * 注意：
   *
   * 1. 本接口不支持并发调用
   * 2.
   * 如果外部指定了Allocator，那么建议Allocator应该与stream绑定，如果出现同一个allocator，匹配不同的stream多次调用Execute接口时，
   *    需要满足两个条件：不可以并发调用，在切换stream执行中间，需要对上一条stream做流同步
   * 3.
   * 若外部指定了Allocator，在模型执行完成前，不可以将Allocator中的内存归还给操作系统（即使这块内存已经由执行器归还给Allocator）
   *
   * @param arg 执行参数
   * @param inputs 网络的输入tensor，从调用本接口开始，到流同步等待本模型执行结束之前，用户需要保证传入的Tensor有效
   * @param input_num 输入tensor的数量
   * @param outputs 网络的输出tensor
   * @param output_num 输出tensor的数量
   * @return 成功时返回`ge::GRAPH_SUCCESS`
   */
  ge::graphStatus Execute(const ModelExecuteArg &arg, Tensor **inputs, size_t input_num, Tensor **outputs,
                          size_t output_num);
  ge::graphStatus ExecuteSync(Tensor **inputs, size_t input_num, Tensor **outputs, size_t output_num);
  ge::graphStatus UnLoad();

  const ModelDesc &GetModelDesc() const;
  void SetModelDesc(ModelDesc *model_desc);
  ExeGraphExecutor *GetExeGraphExecutor(const SubExeGraphType type) {
    if (type >= kSubExeGraphTypeEnd) {
      return nullptr;
    }
    return &graphs_[static_cast<size_t>(type)];
  }
  ExecutorSubscribersScheduler &GetSubscribers();
  uint32_t GetIterationNum() const;
  const ExecutorSubscribersScheduler &GetSubscribers() const;
  ge::graphStatus ArrangeModelLoadArg(const ModelLoadArg &arg, std::vector<void *> &const_inputs);

  ModelV2Executor(const ModelV2Executor &) = delete;
  ModelV2Executor(ModelV2Executor &&) = delete;
  ModelV2Executor &operator=(const ModelV2Executor &) = delete;
  ModelV2Executor &operator=(ModelV2Executor &&) = delete;
  void SetFileConstantWeightDir(const std::string &file_constant_weight_dir) {
    file_constant_weight_dir_ = file_constant_weight_dir;
  }
  const std::string &GetFileConstantWeightDir() const {
    return file_constant_weight_dir_;
  }
  /**
   * @brief 获取aipp相关config info
   * @param [in] index 输入index
   * @param [out] aipp_info 待输出的aipp info
   * @return 成功时返回`ge::SUCCESS`
   */
  ge::Status GetAippInfo(const uint32_t index, ge::AippConfigInfo &aipp_info) const;
  /**
   * @brief 获取aipp输入类型
   * @param [in] index 输入index
   * @param [out] aipp_type 待返回的aipp type
   * @param [out] aipp_index 待返回的aipp index
   * @return 成功时返回`ge::SUCCESS`
   */
  ge::Status GetAippType(const uint32_t index, ge::InputAippType &aipp_type, size_t &aipp_index) const;

  ge::Status GetOriginAippInputInfo(const uint32_t index, ge::OriginInputInfo &orig_aipp_input_info) const;

  ge::Status GetAllAippInputOutputDims(const uint32_t index, std::vector<ge::InputOutputDims> &input_dims,
      std::vector<ge::InputOutputDims> &output_dims) const;

  ge::Status InitAipp(const ge::ComputeGraphPtr &root_graph);

 private:
  friend class ModelV2ExecutorBuilder;
  friend class ModelV2ExecutorTestHelper;
  ModelV2Executor();
  ge::graphStatus SpecifyArgsInputs(const ModelExecuteArg &arg, size_t input_num, ExeGraphExecutor &graph_executor);
  ge::graphStatus OccupyStreamResource(const ModelExecuteArg &arg, TypedContinuousVector<rtStream_t> *&streams,
                                       TypedContinuousVector<rtEvent_t> *&events,
                                       TypedContinuousVector<rtNotify_t> *&notifies);
  ge::Status InitRtVarManager(const ModelLoadArg &load_arg);
  ge::graphStatus CheckIoReuseAddrs(Tensor **inputs, size_t input_num, Tensor **outputs, size_t output_num) const;

 private:
  // to keep host resource live longer than resource_guard_
  // resource guarder may holding pointer from host_resource_center_
  ge::HostResourceCenterPtr host_resource_center_;
  TopologicalResourceGuard resource_guard_;
  std::array<ExeGraphExecutor, kSubExeGraphTypeEnd> graphs_;
  ModelDesc *model_desc_ = nullptr;
  rtStream_t default_stream_ = nullptr;
  ExecutorSubscribersScheduler subscribers_;
  ExecutorState state_ = ExecutorState::kInit;
  std::string file_constant_weight_dir_;
  /*
    * 背景：对于aipp离线推理场景下，acl需要获取编译时期很多aipp的相关信息，rt2场景下需要适配
    * 临时规避方案：因为是客户问题，时间比较紧，所以当前简单仿照静态shape下获取aipp的逻辑，
      数据结构和对应给acl的api保持一致，rt2放在ModelV2Executor类成员变量并提供相应接口
    * 正式方案：对于aipp场景下新定义一个数据结构，例如AllAippInfo的结构体，该结构体里把所有aipp需要的信息都放在里面，
      静态动态下只提供一个API，例如GetAllAippINfo()返回给acl，这样ge只提供一个接口，acl获取完之后相应的Get接口直接在acl层闭环，
      该方案涉及metadef，air,acl三个仓的联合修改。
  */
  // for aipp
  std::map<uint32_t, ge::AippConfigInfo> aipp_info_list_;
  std::map<uint32_t, std::pair<ge::InputAippType, size_t>> aipp_type_list_;
  std::map<uint32_t, ge::OriginInputInfo> orig_aipp_input_info_;
  std::map<uint32_t, std::pair<std::vector<ge::InputOutputDims>, std::vector<ge::InputOutputDims>>> aipp_dims_info_;

  EventAllocator builtin_event_allocator_;
  StreamAllocator builtin_stream_allocator_; // insure stream destroy before event
  NotifyAllocator builtin_notify_allocator_;
  uint64_t load_session_id_{std::numeric_limits<uint64_t>::max()};
  RtSession default_rt_session_{load_session_id_};

  // For output reuse input memory address validation
  std::vector<std::pair<size_t, size_t>> io_same_addr_pairs_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_H_
