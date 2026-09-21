/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_INFO_H_
#define ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_INFO_H_

#include <map>
#include <vector>
#include <memory>
#include "graph/node.h"
#include "tensor_engine/fusion_types.h"
#include "tensor_engine/fusion_constants.h"
#include "tensor_engine/tbe_attr_value.h"
#include "tensor_engine/tbe_op_param.h"

namespace te {
class TbeOpInfo : public std::enable_shared_from_this<TbeOpInfo> {
public:
    TbeOpInfo(const std::string &name,
              const std::string &moduleName,
              const std::string &opType,
              const std::string &engineType)
        : op_name_(name),
          op_module_name_(moduleName),
          op_type_(opType),
          engine_type_(engineType),
          core_type_(engineType),
          op_L1_space_(-1),
          op_unknown_shape_(false),
          op_use_int64_(false),
          is_dynamic_impl_(false),
          build_type_(ACCURATELY_BUILD),
          dynamicRankType_(DynamicRankType::NOT_SUPPORT),
          rangeLimitType_(RangeLimitType::UNLIMITED),
          isSingleOpScene_(false),
          isOriSingleOpScene_(false),
          needPreCompile_(false),
          opJitCompile_(JitCompileType::DEFAULT),
          vectorCoreType_(VectorCoreType::DEFAULT),
          ubFusionSpaceSize_(-1)
    {
    }

    ~TbeOpInfo()
    {
    }

    void GetName(std::string& name) const
    {
        name = op_name_;
    }

    const std::string& GetName() const
    {
        return op_name_;
    }

    void GetRealName(std::string& realName) const
    {
        realName = op_real_name_;
    }

    const std::string& GetRealName() const
    {
        return op_real_name_;
    }

    void SetRealName(const std::string& realName)
    {
        op_real_name_ = realName;
    }

    void GetOpType(std::string &optype) const
    {
        optype = op_type_;
    }

    const std::string& GetOpType() const
    {
        return op_type_;
    }

    void GetOpFileName(std::string& fileName) const
    {
        fileName = op_file_name_;
    }

    const std::string& GetOpFileName() const
    {
        return op_file_name_;
    }

    void SetOpFileName(const std::string &fileName)
    {
        op_file_name_ = fileName;
    }

    void GetOpFuncName(std::string& funcName) const
    {
        funcName = op_func_name_;
    }

    const std::string& GetOpFuncName() const
    {
        return op_func_name_;
    }

    void SetOpFuncName(const std::string &funcName)
    {
        op_func_name_ = funcName;
    }

    const std::string& GetEngineType() const
    {
        return engine_type_;
    }

    void SetOpCoreType(const std::string &coreType)
    {
        core_type_ = coreType;
    }

    void GetOpCoreType(std::string &coreType) const
    {
        coreType = core_type_;
    }

    const std::string& GetOpCoreType() const
    {
        return core_type_;
    }

    void GetCheckSupportedFunc(std::string& checkName) const
    {
        checkName = op_check_supported_name_;
    }

    const std::string& GetCheckSupportedFunc() const
    {
        return op_check_supported_name_;
    }

    void SetCheckSupportedFunc(const std::string &checkName)
    {
        op_check_supported_name_ = checkName;
    }

    /**
     * @brief: normalize function name
     * @param [in] funcName           op function name
     * @return [out] bool             succ or fail for normalizing
     */
    void NormalizeFuncName(std::string& funcName) const;
    bool GetModuleName(std::string& moduleName) const;
    void GetFuncName(std::string& funcName) const;

    void GetOpImplSwitch(std::string& opImplSwitch) const
    {
        opImplSwitch = opImplSwitch_;
    }

    const std::string& GetOpImplSwitch() const
    {
        return opImplSwitch_;
    }

    void SetOpImplSwitch(const std::string& opImplSwitch)
    {
        opImplSwitch_ = opImplSwitch;
    }

    void SetOpsPathNamePrefix(const std::string &opsPathNamePrefix) {
        opsPathNamePrefix_ = opsPathNamePrefix;
    }

    const std::string& GetOpsPathNamePrefix() const
    {
        return opsPathNamePrefix_;
    }

    JitCompileType GetOpJitCompile() const
    {
        return opJitCompile_;
    }

    void SetOpJitCompile(const JitCompileType &jitCompile)
    {
        opJitCompile_ = jitCompile;
    }

    void AddInput(const TbeOpParam& param)
    {
        inputs_.push_back(param);
    }

    void GetInputs(std::vector<TbeOpParam>& inputs) const
    {
        inputs = inputs_;
    }

    const std::vector<TbeOpParam>& GetInputs() const
    {
        return inputs_;
    }

    std::vector<TbeOpParam>& MutableInputs()
    {
        return inputs_;
    }

    void SetInputs(const std::vector<TbeOpParam> &inputs)
    {
        inputs_.assign(inputs.begin(), inputs.end());
    }

    void AddOutput(const TbeOpParam& param)
    {
        outputs_.push_back(param);
    }

    void GetOutputs(std::vector<TbeOpParam> &outputs) const
    {
        outputs = outputs_;
    }

    const std::vector<TbeOpParam>& GetOutputs() const
    {
        return outputs_;
    }

    std::vector<TbeOpParam>& MutableOutputs()
    {
        return outputs_;
    }

    void SetOutputs(const std::vector<TbeOpParam> &outputs)
    {
        outputs_.assign(outputs.begin(), outputs.end());
    }

    void AddAttrValue(const TbeAttrValue& value)
    {
        attr_values_.push_back(value);
    }

    void GetAttrValues(std::vector<TbeAttrValue>& attrValues) const
    {
        attrValues = attr_values_;
    }

    const std::vector<TbeAttrValue>& GetAttrValues() const
    {
        return attr_values_;
    }

    void SetPrivateAttrs(std::vector<TbeAttrValue>& privateAttrValues)
    {
        private_attr_values_ = privateAttrValues;
    }

    const std::vector<TbeAttrValue>& GetPrivateAttrValues() const
    {
        return private_attr_values_;
    }

    void GetPattern(std::string& pattern) const
    {
        pattern = pattern_;
    }

    const std::string& GetPattern() const
    {
        return pattern_;
    }

    void SetPattern(const std::string &pattern)
    {
        pattern_ = pattern;
    }

    const std::string& GetOpImplMode() const
    {
        return op_impl_mode_;
    }

    void SetOpImplMode(const std::string& opImplMode)
    {
        op_impl_mode_ = opImplMode;
    }

    void ClearOpImplMode()
    {
        op_impl_mode_.clear();
    }

    void GetExtraParams(std::string& extraParams) const
    {
        extraParams = extra_params_;
    }

    const std::string& GetExtraParams() const
    {
        return extra_params_;
    }

    void SetExtraParams(const std::string& extraParams)
    {
        extra_params_ = extraParams;
    }

    const std::string& GetHashedExtraParams() const
    {
        return hashed_extra_params_;
    }

    void SetHashedExtraParams(const std::string& hasedExtraParams)
    {
        hashed_extra_params_ = hasedExtraParams;
    }

    bool IsDynamicImpl() const
    {
        return is_dynamic_impl_;
    }

    void SetDynamicImpl(const bool& is_dynamic_impl)
    {
        is_dynamic_impl_ = is_dynamic_impl;
    }

    void GetKernelName(std::string& kernelName) const
    {
        kernelName = kernel_name_;
    }

    const std::string& GetKernelName() const
    {
        return kernel_name_;
    }

    void SetKernelName(const std::string &kernelName)
    {
        kernel_name_ = kernelName;
    }

    void GetL1Space(int64_t& l1Space) const
    {
        l1Space = op_L1_space_;
    }

    int64_t GetL1Space() const
    {
        return op_L1_space_;
    }

    void SetL1Space(const int64_t l1Space)
    {
        op_L1_space_ = l1Space;
    }

    void SetIsUnknownShape(const bool unknown)
    {
        op_unknown_shape_ = unknown;
    }

    bool GetIsUnknownShape() const
    {
        return op_unknown_shape_;
    }

    void SetFlagUseInt64(const bool opUseInt64)
    {
        op_use_int64_ = opUseInt64;
    }

    bool GetFlagUseInt64() const
    {
        return op_use_int64_;
    }

    void GetOptions(std::map<std::string, std::string>& options) const
    {
        options = options_;
    }

    const std::map<std::string, std::string>& GetOptions() const
    {
        return options_;
    }

    void SetOptions(const std::map<std::string, std::string> &options)
    {
        options_ = options;
    }

    bool GetOption(const std::string& key, std::string &value) const
    {
        auto optionsFind = options_.find(key);
        if (optionsFind != options_.end()) {
            value = optionsFind->second;
            return true;
        }
        return false;
    }

    void SetOption(const std::string& key, const std::string &value)
    {
        options_[key] = value;
    }

    void SetBuildType(BUILD_TYPE build_type)
    {
        build_type_ = build_type;
    }

    BUILD_TYPE GetBuildType() const
    {
        return build_type_;
    }

    void SetRangeLimitType(const RangeLimitType &rangeLimitType)
    {
        rangeLimitType_ = rangeLimitType;
    }

    RangeLimitType GetRangeLimitType() const
    {
        return rangeLimitType_;
    }

    void SetNode(const ge::NodePtr &nodePtr)
    {
        nodePtr_ = nodePtr;
    }

    void GetNode(ge::NodePtr &nodePtr) const
    {
        nodePtr = nodePtr_;
    }

    const ge::NodePtr& GetNode() const
    {
        return nodePtr_;
    }

    bool IsSupportDynamicRank() const
    {
        return dynamicRankType_ != DynamicRankType::NOT_SUPPORT;
    }

    DynamicRankType GetDynamicRankType() const
    {
        return dynamicRankType_;
    }

    void SetDynamicRankType(const DynamicRankType &dynamicRankType)
    {
        dynamicRankType_ = dynamicRankType;
    }

    void SetOpStorePattern(const std::string &opStorePattern)
    {
        opStorePattern_ = opStorePattern;
    }

    void GetOpStorePattern(std::string &opStorePattern) const
    {
        opStorePattern = opStorePattern_;
    }

    const std::string& GetOpStorePattern() const
    {
        return opStorePattern_;
    }

    void SetPrebuildPattern(const std::string &prebuildPattern)
    {
        prebuildPattern_ = prebuildPattern;
    }

    void GetPrebuildPattern(std::string &prebuildPattern) const
    {
        prebuildPattern = prebuildPattern_;
    }

    const std::string& GetPrebuildPattern() const
    {
        return prebuildPattern_;
    }

    const std::string& GetOpDebugConfig() const
    {
        return opDebugConfig_;
    }

    void SetOpDebugConfig(const std::string &opDebugConfig)
    {
        opDebugConfig_ = opDebugConfig;
    }

    void SetSingleOpSceneFlag(const bool &isSingleOpScene)
    {
        isSingleOpScene_ = isSingleOpScene;
    }

    void GetSingleOpSceneFlag(bool &isSingleOpScene) const
    {
        isSingleOpScene = isSingleOpScene_;
    }

    bool IsSingleOpScene() const
    {
        return isSingleOpScene_;
    }

    void SetOriSingleOpSceneFlag(const bool &isOriSingleOpScene)
    {
        isOriSingleOpScene_ = isOriSingleOpScene;
    }

    void GetOriSingleOpSceneFlag(bool &isOriSingleOpScene) const
    {
        isOriSingleOpScene = isOriSingleOpScene_;
    }

    bool IsOriSingleOpScene() const
    {
        return isOriSingleOpScene_;
    }

    bool IsNeedPreCompile() const
    {
        return needPreCompile_;
    }

    void SetNeedPreCompile(const bool needPreCompile)
    {
        needPreCompile_ = needPreCompile;
    }

    VectorCoreType GetVectorCoreType() const
    {
        return vectorCoreType_;
    }

    void SetVectorCoreType(const VectorCoreType vectorCoreType)
    {
        vectorCoreType_ = vectorCoreType;
    }

    void SetCustAicNum(const std::string &custAicNum)
    {
        cust_aic_num_ = custAicNum;
    }
 
    const std::string& GetCustAicNum() const
    {
        return cust_aic_num_;
    }
 
    void SetCustAivNum(const std::string &custAivNum)
    {
        cust_aiv_num_ = custAivNum;
    }
 
    const std::string& GetCustAivNum() const
    {
        return cust_aiv_num_;
    }

    int64_t GetUBSpaceSize() const
    {
        return ubFusionSpaceSize_;
    }

    void SetUBSpaceSize(const int64_t size)
    {
        ubFusionSpaceSize_ = size;
    }

    bool operator==(TbeOpInfo& rObject);

private:
    // need to adapt operator== func while adding new variable
    // parameter from upper
    std::string op_name_;
    std::string op_real_name_;
    std::string op_module_name_;
    std::string op_type_;
    std::string engine_type_;
    std::string op_file_name_;
    std::string op_func_name_;
    std::string op_check_supported_name_;
    std::string core_type_;
    std::string opImplSwitch_;
    std::string opsPathNamePrefix_;

    // parameter from lower
    std::string kernel_name_;
    std::string pattern_;
    std::string op_impl_mode_;
    std::string extra_params_;
    std::string hashed_extra_params_;

    // L1 fusion parameter
    int64_t op_L1_space_;
    bool op_unknown_shape_;
    bool op_use_int64_;
    bool is_dynamic_impl_;
    BUILD_TYPE build_type_;

    // op parameter
    std::vector<TbeOpParam> inputs_;
    std::vector<TbeOpParam> outputs_;
    std::vector<TbeAttrValue> attr_values_;
    std::vector<TbeAttrValue> private_attr_values_;

    // ge options args
    std::map<std::string, std::string> options_;

    DynamicRankType dynamicRankType_;
    RangeLimitType rangeLimitType_;
    std::string opStorePattern_;
    std::string prebuildPattern_;
    std::string opDebugConfig_;
    // whether there is only one node in sub graph.
    bool isSingleOpScene_;
    // whether there is only one node in original graph, such as in acl mode.
    bool isOriSingleOpScene_;
    bool needPreCompile_;
    JitCompileType opJitCompile_;
    VectorCoreType vectorCoreType_;
    ge::NodePtr nodePtr_{nullptr};
    int64_t ubFusionSpaceSize_;
    std::string cust_aic_num_;
    std::string cust_aiv_num_;
};
using TbeOpInfoPtr = std::shared_ptr<TbeOpInfo>;
using ConstTbeOpInfoPtr = std::shared_ptr<const TbeOpInfo>;
}
#endif  // ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_INFO_H_
