/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYTHON_ADAPTER_MANAGER_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYTHON_ADAPTER_MANAGER_H_

#include "inc/te_fusion_types.h"
#include "inc/te_fusion_log.h"
#include "python_adapter/py_wrapper.h"
#include "python_adapter/python_api_call.h"

namespace te {
namespace fusion {
class PythonAdapterManager {
public:
    static PythonAdapterManager &Instance();
    bool Initialize();
    bool Finalize();
    bool IsInitParallelCompilation() const;
    bool GetFinishedCompilationTask(const uint64_t graphId, std::vector<OpBuildTaskResultPtr> &taskRetVec) const;

    bool DispatchBinarySingleOpCompileTask(const std::string &taskDescStr) const;
    bool DispatchBinaryFusionOpCompileTask(const std::string &taskDescStr) const;
    bool CallGcCollect();

    template<typename... Args>
    bool BuildOpAsync(const OpBuildTaskPtr &task, const char *asyncFunc, const char *argFormat, Args... args) const
    {
        uint64_t graphId = task->graphId;
        uint64_t taskId = task->taskId;
        int64_t l1Size = PythonApiCall::Instance().GetL1SpaceSize();

        std::string format = argFormat;

        format = "kkl" + format;
        PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleCompileTaskMgr_,
                                                  asyncFunc, format.c_str(), graphId, taskId, l1Size, args...);
        if (pyRes == nullptr) {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to call [%s]!", asyncFunc);
            task->status = OP_TASK_STATUS::OP_TASK_FAIL;
            return false;
        }

        TE_DBGLOG("Python call %s succeed!", asyncFunc);
        task->status = OP_TASK_STATUS::OP_TASK_PENDING;
        return true;
    }
private:
    PythonAdapterManager() : isInit_(false), isParallelCompileInit_(false) {}
    ~PythonAdapterManager() {}
    static bool InitCannKB();
    bool InitPyModuleAndApiCall();
    bool ResetPyModule();
    bool InitParallelCompilation();
    static bool GetPidTimestamp(std::string &pidTimestamp);
    static void ParallelCompilerProcessesNumCheck();
    static OpBuildTaskResultPtr ParseOpTaskResult(PyObject *pyRes);
    static bool ParseResult(const char *opRes, int typeValue, std::string &jsonFilePath, std::string &pattern,
                            std::string &coreType, std::string &compileInfo, std::string &compileInfoKey,
                            int &fusionCheckResCode);
    static bool GetCompileInfo(const std::string &jsonFilePath, std::string &compileInfo, std::string &compileInfoKey);
    static void PyErrMessageReport(PyObject *pyErrMsgTuple);

private:
    bool isInit_;
    bool isParallelCompileInit_;
    PyObjectPtr pyModuleFusionMgr_;
    PyObjectPtr pyModuleFusionUtil_;
    PyObjectPtr pyModuleCcePolicy_;
    PyObjectPtr pyModuleCompileTaskMgr_;
    PyObjectPtr pyModuleParallCompile_;
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYTHON_ADAPTER_MANAGER_H_
