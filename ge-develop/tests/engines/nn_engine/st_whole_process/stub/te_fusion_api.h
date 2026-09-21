/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TEST_ENGINE_NNENG_DEPENDS_TE_FUSION_INCLUDE_TE_FUSION_API_H_
#define AIR_TEST_ENGINE_NNENG_DEPENDS_TE_FUSION_INCLUDE_TE_FUSION_API_H_

#include <cstring>
#include "tensor_engine/fusion_api.h"

namespace te {
class TeStubApi {
 public:
  virtual ~TeStubApi() = default;

  virtual bool TbeInitialize(const std::map<std::string, std::string> &options, bool *isSupportParallelCompilation);


  virtual bool TbeFinalize();

  virtual bool WaitAllFinished(uint64_t graphId, vector<FinComTask> &tasks);

  virtual bool CheckTbeSupported(TbeOpInfo &opinfo, CheckSupportedResult &isSupport, std::string &reason);

  virtual bool CheckOpSupported(TbeOpInfo& opinfo, CheckSupportedInfo &result);

  virtual bool PreBuildTbeOp(TbeOpInfo &opinfo, uint64_t taskId, uint64_t graphId);

  virtual OpBuildResCode TeFusion(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                  const std::vector<ge::NodePtr> &toBeDel,
                                  uint64_t taskId, uint64_t graphId, const std::string &strategy);

  virtual OpBuildResCode TeFusionV(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                   const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId, uint64_t graphId,
                                   uint64_t sgtThreadIndex, const std::string &strategy);

  virtual bool SelectTbeOpFormat(const TbeOpInfo &tbeOpInfo, std::string &opDtypeFormat);

  virtual bool GetOpUniqueKeys(const TbeOpInfo &tbeOpInfo, std::vector<std::string> &opUniqueKeys);

  virtual bool TeFusionEnd();

  virtual LX_QUERY_STATUS GetOpInfo(const TbeOpInfo &tbeOpInfo, std::string &result);

  virtual OpBuildResCode FuzzBuildTbeOp(uint64_t taskId, uint64_t graphId, ge::Node &node);

  virtual bool SetFuzzyBuildSupportInfoToNode(ge::NodePtr &nodePtr, const std::string &supportInfoStr,
                                              const std::vector<size_t> &indirectIndexs,
                                              const std::vector<std::pair<size_t, size_t>> &directIndexMap);

  virtual bool CheckIsTbeGeneralizeFuncRegistered(const TbeOpInfo &tbeOpInfo, bool &hasRegisteredFunc);

  virtual bool TeGeneralize(const TbeOpInfo &tbeOpInfo, const TE_GENERALIZE_TYPE &generalize_type,
                            const ge::NodePtr &nodePtr);

  virtual bool GetOpSpecificInfo(const TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo);

  virtual bool DynamicShapeRangeCheck(const TbeOpInfo &tbeOpInfo, bool &isSupported,
                                      std::vector<size_t> &upperLimitedInputIndexs,
                                      std::vector<size_t> &lowerLimitedInputIndexs);

  virtual bool QueryOpPattern(const std::vector<std::pair<std::string, std::string>>& opPatternVec);
  virtual OpBuildResCode BuildSuperKernel(const std::vector<ge::Node *> &graphNodes, ge::OpDescPtr opDesc,
    const uint64_t taskId, const uint64_t graphId);
  virtual std::string GetKernelMetaDir();
  std::map<uint64_t, vector<FinComTask>> tasks_;
};

class TeStub {
 public:
  static TeStub &GetInstance() {
    static TeStub instance;
    return instance;
  }

  void SetImpl(const std::shared_ptr<TeStubApi> &impl) {
    impl_ = impl;
  }

  TeStubApi *GetImpl() {
    return impl_.get();
  }

  void Reset() {
    impl_ = std::make_shared<TeStubApi>();
  }
  void* real_prebuild_func_;
  void* wait_all_finished_;
  void* tbe_initialize_;
  void* te_fusion_;
  void* te_fusion_v_;
 private:
  TeStub() : impl_(std::make_shared<TeStubApi>()) {
  }
  std::shared_ptr<TeStubApi> impl_;
};

class RealTimeCompilation : public te::TeStubApi {
 public:
  virtual ~RealTimeCompilation() = default;
  virtual te::OpBuildResCode TeFusionV(std::vector<ge::Node *> teGraphNode, ge::OpDescPtr opDesc,
                                       const std::vector<ge::NodePtr> &toBeDel, uint64_t taskId, uint64_t graphId,
                                       uint64_t sgtThreadIndex, const std::string &strategy) override {
    auto te_fusion_v_func =
        (te::OpBuildResCode(*)(std::vector<ge::Node *>, ge::OpDescPtr, const std::vector<ge::NodePtr> &,
                               uint64_t, uint64_t, uint64_t,
                               const std::string &)) (te::TeStub::GetInstance().te_fusion_v_);
    return te_fusion_v_func(teGraphNode, opDesc, toBeDel, taskId, graphId, sgtThreadIndex, strategy);
  }

  virtual bool PreBuildTbeOp(TbeOpInfo &opinfo, uint64_t taskId, uint64_t graphId) override {
    const char *dump_ge_graph = std::getenv("TE_PARALLEL_COMPILER");
    if (dump_ge_graph == nullptr || strcmp(dump_ge_graph, "-1") != 0) {
      auto prebuild_func_ =
          (bool(*)(te::TbeOpInfo &, uint64_t, uint64_t))(te::TeStub::GetInstance().real_prebuild_func_);
      return prebuild_func_(opinfo, taskId, graphId);
    } else {
      // 在本地windows上的ubuntu环境执行时会遇到socket相关的protocol不支持的问题，暂时无法解决。
      // 所以在本地环境上采用TE_PARALLEL_COMPILER=-1的形式来表达当前需要串行编译。
      return te::TeStubApi::PreBuildTbeOp(opinfo, taskId, graphId);
    }
  }
};
}
#endif //AIR_TEST_ENGINE_NNENG_DEPENDS_TE_FUSION_INCLUDE_TE_FUSION_API_H_
