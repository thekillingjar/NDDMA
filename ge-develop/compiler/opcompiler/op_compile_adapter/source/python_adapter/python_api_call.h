/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYTHON_API_CALL_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYTHON_API_CALL_H_

#include <Python.h>
#include <map>
#include <string>
#include <mutex>
#include <nlohmann/json.hpp>

#include "graph/node.h"
#include "python_adapter/py_wrapper.h"
#include "tensor_engine/tbe_op_info.h"

namespace te {
namespace fusion {
enum class ERR_CODE {
    ERR_FAIL = -2,
    ERR_WARN = -1,
    NO_ERR = 0,
};

class PythonApiCall {
public:
    static PythonApiCall &Instance();
    // lock for singleton mode
    static std::mutex mtx_;

    void PythonApiCallInit(PyObjectPtr &pyModuleFusionMgr, PyObjectPtr &pyModuleFusionUtil,
                           PyObjectPtr &pyModuleCompileTaskMgr, PyObjectPtr &pyModuleCcePolicy);

    /*
     * @brief: check tbe op capability
     * @param [in] opinfo: op total parameter set
     * @param [out] IsSupport: result to support or not
     * @return bool: check process ok or not
     */
    bool CheckSupported(const TbeOpInfo &opinfo, CheckSupportedResult &isSupport, std::string &reason);

    /**
     * @brief select tbe op format
     * @param [in] tbeOpInfo            op info
     * @param [out] opDtypeformat       op date format
     * @return [out] bool               succ or fail
     */
    bool SelectTbeOpFormat(const TbeOpInfo &opinfo, std::string &opDtypeFormat);

    LX_QUERY_STATUS GetOpInfo(const TbeOpInfo &tbeOpInfo, std::string &result);

    bool CheckIsTbeGeneralizeFuncRegistered(const TbeOpInfo &tbeOpInfo, bool &hasRegisteredFunc);

    bool GetCheckAndGeneralizedResFromTbe(const TbeOpInfo &tbeOpInfo, const string &buildType,
                                          nlohmann::json &generalizedRes);

    bool DynamicShapeRangeCheck(const TbeOpInfo &tbeOpInfo, bool &isSupported,
                                std::vector<size_t> &upperLimitedInputIndexs,
                                std::vector<size_t> &lowerLimitedInputIndexs);

    bool GetOpSpecificInfo(const TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo);

    bool GetOpUniqueKeys(const TbeOpInfo &tbeOpInfo, const std::string &jsonStr,
                         std::vector<std::string> &opUniqueKeyList);

    bool IsOpImplModeSupported(const ConstTbeOpInfoPtr &tbeOpInfoPtr, bool &isSupported);

    bool UpdateSingleOpModule(const TbeOpInfo &tbeOpInfo, std::string &opModule, std::string *pyModulePath = nullptr);

    bool GetAttrInNode(const TbeOpInfoPtr &pOpInfo, const std::string &jsonStr, std::string &opArgs);

    bool GenerateStrSha256HashValue(const std::string &str, std::string &result) const;

    bool GetBinFileSha256Value(const char *binData, const size_t binSize, std::string &sha256Val) const;

    int64_t GetL1SpaceSize() const;
    bool SetL1SpaceSize(const int64_t l1SpaceValue);
    void ResetL1SpaceSize();
    /*
     * @brief: config op common cfg, like, ddk version, module and opname
     * @param [in] opinfo: op info, include op name, module name, parameters, and so on
     * @return bool: save compute parameter to python ok or not
     */
    bool SetOpParams(const TbeOpInfo &opInfo, PyObject *pyOutputs);

    bool DumpFusionJson(const std::string &jsonStr, const std::string &dumpPath) const;

    bool SyncOpTuneParams() const;
private:
    PythonApiCall();
    ~PythonApiCall();
    template<typename... Args>
    bool CallPyFuncWithTbeOpInfo(const TbeOpInfo &tbeOpInfo, PyObjectPtr &pyRes, const bool &isSingleOpBuild,
                                 const std::string &pyFunc, const std::string &argFormat, Args... args);

    bool ImportOpFilePath(std::string &pathName);

    /**
      * @brief: normalize module name
      * @param [in] ModuleName: original module name
      * @return bool: normaliable result
      */
    bool NormalizeModuleName(std::string &ModuleName, std::string *pyModulePath);

    bool ParsePyResultOfString(const PyObjectPtr &pyRes, std::string &resString) const;

    bool ParsePyResultOfListString(const PyObjectPtr &pyRes, std::vector<std::string> &resList) const;

    bool CheckFuncInPyModule(const std::string &opFuncName, const std::string &opModule) const;

    void ParseCheckResult(PyObject *resPtr, const std::string &opModule,
                          CheckSupportedResult &isSupport, std::string &reason) const;

    bool ParseDynamicShapeRangeCheckFromTbe(nlohmann::json &generalizedRes, bool &isSupported,
                                            std::vector<size_t> &upperLimitedInputIndexs,
                                            std::vector<size_t> &lowerLimitedInputIndexs) const;

    PyObject *GetStrDictPyObj(const std::map<std::string, std::string> &infoMap) const;

    PyObject *GetGeneralizeConfig(const TbeOpInfo &tbeOpInfo, const string &buildType) const;

    ERR_CODE CallOpFunc(const TbeOpInfo &opinfo, const std::string &opFuncName,
                        const bool &isNeedJudgeOpFuncName, PyObjectPtr &pyRes);
    PyObjectPtr pyModuleFusionMgr_;
    PyObjectPtr pyModuleCompileTaskMgr_;
    PyObjectPtr pyModuleFusionUtil_;
    PyObjectPtr pyModuleCcePolicy_;

    int64_t l1SpaceSize_;
    std::set<std::string> pySysPath_;
    mutable std::mutex pySysPathMutex_;
};
} // namespace fusion
} // namespace te
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYTHON_API_CALL_H_
