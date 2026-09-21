/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "python_adapter/python_adapter_manager.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/te_file_utils.h"
#include "common/te_config_info.h"
#include "dfxinfo_manager/trace_utils.h"
#include "python_adapter/py_decouple.h"
#include "python_adapter/py_object_utils.h"
#include "python_adapter/python_api_call.h"
#include "common/cann_kb_adapter.h"

namespace te {
namespace fusion {
namespace {
constexpr int PARALLEL_PROCESSES_NUM_MIN = 1;
constexpr int PARALLEL_PROCESSES_NUM_MAX = 32;
const std::string kInnerErrType = "2";
}
PythonAdapterManager &PythonAdapterManager::Instance()
{
    static PythonAdapterManager pythonAdapterManager;
    return pythonAdapterManager;
}

bool PythonAdapterManager::Initialize()
{
    if (isInit_) {
        TE_DBGLOG("Python adapter manager has already been initialized.");
        return true;
    }

    TE_DBGLOG("Begin to initialize python adapter manager.");
    if (!HandleManager::Instance().Initialize()) {
        REPORT_TE_INNER_ERROR("Failed to initialize handle manager.");
        return false;
    }

    if (!TeConfigInfo::Instance().IsDisableOpCompile()) {
        // need InitCannKB before get PyLockGIL
        if (!InitCannKB()) {
            REPORT_TE_INNER_ERROR("AOE failed to call InitCannKB.");
            return false;
        }
        TraceUtils::SubmitGlobalTrace("CannKb has been initialized.");
    }

    // get gil lock
    PyLockGIL pyLockGIL;
    if (!PyWrapper::InitPyLogger()) {
        REPORT_TE_INNER_ERROR("Failed to initialize py logger.");
        return false;
    }

    if (!InitPyModuleAndApiCall()) {
        REPORT_TE_INNER_ERROR("Failed to initialize Python module and API call.");
        return false;
    }

    isParallelCompileInit_ =
        !TeConfigInfo::Instance().IsDisableUbFusion() || !TeConfigInfo::Instance().IsDisableOpCompile();
    if (isParallelCompileInit_) {
        TE_FUSION_TIMECOST_START(InitParalCompilation);
        if (!InitParallelCompilation()) {
            TE_ERRLOG("Failed to initialize parallel compilation.");
            return false;
        }
        TE_FUSION_TIMECOST_END(InitParalCompilation, "ParallelCompilation::Initialize");
        TraceUtils::SubmitGlobalTrace("Parallel Compilation has been initialized.");
    }
    TE_INFOLOG("Python adapter manager has been initialized successfully.");
    isInit_ = true;
    return true;
}

bool PythonAdapterManager::InitCannKB()
{
    TE_FUSION_TIMECOST_START(InitCannKB);
    std::map<std::string, std::string> sysConfig;
    std::map<std::string, std::string> loadConfig;
    sysConfig.insert(std::pair<std::string, std::string> ("soc_version", TeConfigInfo::Instance().GetSocVersion()));
    sysConfig.insert(std::pair<std::string, std::string> ("core_num", TeConfigInfo::Instance().GetAiCoreNum()));
    loadConfig.insert(std::pair<std::string, std::string> ("op_bank_path", TeConfigInfo::Instance().GetOpBankPath()));
    auto cann_kb_utils = fe::CannKBUtils::Instance();
    if(!cann_kb_utils.InitCannKb()) {
        TE_ERRLOG("Op_compile_adapter failed to call initCannKb.");
        return false;
    }
    fe::CannKb::CANN_KB_STATUS cannKbStatus = cann_kb_utils.CannKbInit(sysConfig, loadConfig);
    if (cannKbStatus != fe::CannKb::CANN_KB_STATUS::CANN_KB_SUCC) {
        TE_ERRLOG("Call CannKbInit failed. Result = [%d]. Init parameters: [%s, %s, %s]", cannKbStatus,
                  TeConfigInfo::Instance().GetSocVersion().c_str(), TeConfigInfo::Instance().GetAiCoreNum().c_str(),
                  TeConfigInfo::Instance().GetOpBankPath().c_str());
        return false;
    }
    TE_FUSION_TIMECOST_END(InitCannKB, "CannKB::Initialize");
    TE_INFOLOG("CannKbInit succeeded. socinfo: [%s, %s, %s].", TeConfigInfo::Instance().GetSocVersion().c_str(),
               TeConfigInfo::Instance().GetAiCoreNum().c_str(), TeConfigInfo::Instance().GetOpBankPath().c_str());
    return true;
}

bool PythonAdapterManager::InitPyModuleAndApiCall()
{
    pyModuleFusionMgr_ = PyWrapper::ImportModule("te_fusion.fusion_manager");
    TE_FUSION_NOTNULL(pyModuleFusionMgr_);
    pyModuleFusionUtil_ = PyWrapper::ImportModule("te_fusion.fusion_util");
    TE_FUSION_NOTNULL(pyModuleFusionUtil_);
    pyModuleCcePolicy_ = PyWrapper::ImportModule("te.platform.cce_policy");
    TE_FUSION_NOTNULL(pyModuleCcePolicy_);
    pyModuleCompileTaskMgr_ = PyWrapper::ImportModule("te_fusion.compile_task_manager");
    TE_FUSION_NOTNULL(pyModuleCompileTaskMgr_);

    TE_DBGLOG("Import fusion_manager/fusion_util/compile_task_manager/te.platform.cce_conf successfully.");

    PythonApiCall::Instance().PythonApiCallInit(pyModuleFusionMgr_, pyModuleFusionUtil_,
                                                pyModuleCompileTaskMgr_, pyModuleCcePolicy_);
    TraceUtils::SubmitGlobalTrace("PythonApiCall has been initialized.");
    return true;
}

bool PythonAdapterManager::GetPidTimestamp(std::string &pidTimestamp)
{
    std::time_t now = std::time(nullptr);
    std::tm nowTm;
    std::tm* tempTm = std::localtime(&now);
    if (tempTm == nullptr) {
        REPORT_TE_INNER_ERROR("Failed to get local time.");
        return false;
    }
    nowTm = *tempTm;

    char ts[TIMESTAMP_LEN];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &nowTm);

    struct timeval tv;
    (void)gettimeofday(&tv, nullptr);
    pidTimestamp = ts + std::to_string(static_cast<int>(tv.tv_usec/1000)) +
                   "_pid" + std::to_string(static_cast<int>(getpid()));
    return true;
}

bool PythonAdapterManager::CallGcCollect() {
    TE_INFOLOG("CallGcCollect");
    PyLockGIL pyLockGIL;
    PyObjectPtr pyCheckRes = PyWrapper::CallMethod(pyModuleFusionUtil_, "gc_collect", nullptr);
    TE_FUSION_CHECK(pyCheckRes == nullptr || HandleManager::Instance().TE_PyObject_IsTrue(pyCheckRes.get()) == 0, {
        REPORT_TE_INNER_ERROR("Failed to call python function gc_collect.");
        return false;
    });
    return true;
}

bool PythonAdapterManager::InitParallelCompilation()
{
    TE_INFOLOG("Starting parallel compilation initialization.");
    ParallelCompilerProcessesNumCheck();
    // check multi process
    PyObjectPtr pyCheckRes = PyWrapper::CallMethod(pyModuleFusionUtil_, "multi_process_check", nullptr);
    TE_FUSION_CHECK(pyCheckRes == nullptr || HandleManager::Instance().TE_PyObject_IsTrue(pyCheckRes.get()) == 0, {
        REPORT_TE_INNER_ERROR("Failed to call python function multi_process_check.");
        return false;
    });
    TraceUtils::SubmitGlobalTrace("Multi process has been checked.");

    pyModuleParallCompile_ = PyWrapper::ImportModule("te_fusion.parallel_compilation");
    if (pyModuleParallCompile_ == nullptr) {
        TE_ERRLOG("Failed to import python module te_fusion.parallel_compilation");
        return false;
    }
    TE_DBGLOG("Import [te_fusion.parallel_compilation] success.");

    std::string pidTimestamp;
    if (!GetPidTimestamp(pidTimestamp)) {
        return false;
    }
    TE_INFOLOG("Pid time stamp %s.", pidTimestamp.c_str());

    PyObject *pySocInfo = PyObjectUtils::GenPySocInfo();
    AUTO_PY_DECREF(pySocInfo);
    PyObject *pyOptionDict = PyObjectUtils::GenPyOptionDict();
    AUTO_PY_DECREF(pyOptionDict);
    int enableEvent = 0;
    int globalLogLevel = dlog_getlevel(TEFUSION, &enableEvent);

    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleParallCompile_, "init_multi_process_env", "iOOiis",
                                              static_cast<int>(!HandleManager::Instance().IsPyEnvInitBeforeTbe()),
                                              pySocInfo, pyOptionDict,
                                              globalLogLevel, enableEvent, pidTimestamp.c_str());
    if (pyRes == nullptr) {
        REPORT_TE_INNER_ERROR("Failed to call function init_multi_process_env");
        return false;
    }

    int processCount = 0;
    int res = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "i", &processCount);
    if (res == 0) {
        REPORT_TE_INNER_ERROR("Failed to parse return of function init_multi_process_env");
        return false;
    }

    if (processCount <= 0) {
        TE_ERRLOG("Process count[%d] is invalid.", processCount);
        return false;
    }

    TE_INFOLOG("Parallel compilation init success. process count: %d.", processCount);
    return true;
}

void PythonAdapterManager::ParallelCompilerProcessesNumCheck()
{
    const std::string &processesNumStr = TeConfigInfo::Instance().GetEnvTeParallelCompiler();
    if (!processesNumStr.empty()) {
        std::map<std::string, std::string> mapArgs = {{"invalid_value", processesNumStr},
                                                      {"argument", "TE_PARALLEL_COMPILER"},
                                                      {"valid_range", "[1, 32]"}, {"default_value", "8"}};
        for (const char &ch : processesNumStr) {
            if (ch < '0' || ch > '9') {
                TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, mapArgs);
                return;
            }
        }
        int count = 0;
        try {
            count = atoi(processesNumStr.c_str());
        } catch (...) {
            TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, mapArgs);
            return;
        }
        if (count < PARALLEL_PROCESSES_NUM_MIN || count > PARALLEL_PROCESSES_NUM_MAX) {
            TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, mapArgs);
        }
    }
}

bool PythonAdapterManager::Finalize()
{
    if (!isInit_) {
        TE_DBGLOG("Python adapter manager has not been initialized.");
        return true;
    }
    if (!TeConfigInfo::Instance().IsDisableOpCompile()) {
        if(!fe::CannKBUtils::Instance().InitCannKb()) {
            TE_ERRLOG("Op_compile_adapter failed to call initCannKb.");
            return false;
        }
        fe::CannKb::CANN_KB_STATUS cannKbStatus = fe::CannKBUtils::Instance().CannKbFinalize();
        if (cannKbStatus != fe::CannKb::CANN_KB_STATUS::CANN_KB_SUCC) {
            TE_ERRLOG("call CannKbFinalize failed. res = [%d].", cannKbStatus);
        } else {
            TE_INFOLOG("cann_kb_finalize success.");
        }
        TraceUtils::SubmitGlobalTrace("CannKb has been finalized.");
    }

    if (!ResetPyModule()) {
        return false;
    }

    if (!HandleManager::Instance().Finalize()) {
        return false;
    }
    isInit_ = false;
    TE_INFOLOG("Python adapter manager has been finalized successfully.");
    return true;
}

bool PythonAdapterManager::ResetPyModule()
{
    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleFusionMgr_, "clear_fusion_params", nullptr);
    if (pyRes == nullptr) {
        TE_WARNLOG("Failed to call function clear_fusion_params.");
    } else {
        TE_INFOLOG("clear_fusion_params success.");
    }
    pyModuleFusionUtil_.reset();
    pyModuleFusionMgr_.reset();
    pyModuleCcePolicy_.reset();
    pyModuleCompileTaskMgr_.reset();
    if (isParallelCompileInit_) {
        pyRes = PyWrapper::CallMethod(pyModuleParallCompile_, "deinit_multi_process_env", nullptr);
        if (pyRes == nullptr) {
            TE_WARNLOG("Failed to call function deinit_multi_process_env");
            return false;
        }
        pyModuleParallCompile_.reset();
    }
    TE_INFOLOG("Python module has been reset successfully.");
    TraceUtils::SubmitGlobalTrace("Python module has been reset.");
    return true;
}

bool PythonAdapterManager::IsInitParallelCompilation() const { return isParallelCompileInit_; }

bool PythonAdapterManager::GetFinishedCompilationTask(const uint64_t graphId,
                                                      std::vector<OpBuildTaskResultPtr> &taskRetVec) const
{
    TE_DBGLOG("Begin to get finished compile tasks of graph[%ld].", graphId);
    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleParallCompile_, "get_finished_compilation_task", "k", graphId);
    if (pyRes == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to call get_finished_compilation_task");
        return false;
    }
    PyObject *pFinishedTaskList = pyRes.get();
    ssize_t finishTaskSize = HandleManager::Instance().TE_PyList_Size(pFinishedTaskList);
    TE_DBGLOG("Finished task size of graph[%ld] is [%zd].", graphId, finishTaskSize);
    for (ssize_t i = 0; i < finishTaskSize; ++i) {
        PyObject *taskItem = HandleManager::Instance().TE_PyList_GetItem(pFinishedTaskList, i);
        TE_FUSION_CHECK_WITH_DUMP_PYERR(taskItem == nullptr, {
            TE_ERRLOG("Failed to get finished task item.");
            return false;
        });

        OpBuildTaskResultPtr taskResPtr = ParseOpTaskResult(taskItem);
        if (taskResPtr == nullptr) {
            TE_ERRLOG("Parse build task result error");
            return false;
        }
        taskRetVec.emplace_back(taskResPtr);
    }
    TE_DBGLOG("Task result size for graph [%ld] is [%zu].", graphId, taskRetVec.size());
    return true;
}

OpBuildTaskResultPtr PythonAdapterManager::ParseOpTaskResult(PyObject *pyRes)
{
    bool invalidInput = (pyRes == nullptr) || (PyDict_Check(pyRes) == 0);
    if (invalidInput) {
        TE_ERRLOG("pyRes is empty!");
        return nullptr;
    }

    int typeValue;
    uint64_t gidValue;
    uint64_t tidValue;
    int statusValue;
    char *resValue = nullptr;
    char *prebuiltOptions = nullptr;
    char *msgValue = nullptr;

    char *errArgsValue = nullptr;
    char *pyExceptMsgValue = nullptr;
    PyObject *pyErrMsgTuple = nullptr;
    char *opRes = nullptr;
    int fusionCheckResCode = CHECK_RES_INITIAL_VALUE;
    using ResDescType = std::tuple<std::string, std::string, bool>;

    std::map<ResDescType, void *> items {
        {ResDescType{"type",             "i", true}, &typeValue}, // true means mandatory
        {ResDescType{"graph_id",         "k", true}, &gidValue},
        {ResDescType{"task_id",          "k", true}, &tidValue},
        {ResDescType{"status_code",      "i", true}, &statusValue},
        {ResDescType{"result",           "s", true}, &resValue},
        {ResDescType{"prebuilt_options", "s", false}, &prebuiltOptions},
        {ResDescType{"info_msg",         "s", true}, &msgValue},
        {ResDescType{"err_args",         "s", false}, &errArgsValue},
        {ResDescType{"except_msg",       "s", false}, &pyExceptMsgValue},
        {ResDescType{"except_tuple_msg", "O", false}, &pyErrMsgTuple},
        {ResDescType{"op_res",           "s", false}, &opRes},
    };
    for (auto &item : items) {
        auto &key = std::get<0>(item.first);
        auto &pyType = std::get<1>(item.first);
        bool mandatory = std::get<2>(item.first);
        auto &value = item.second;
        PyObject *pyValue = HandleManager::Instance().TE_PyDict_GetItemString(pyRes, key.c_str());
        if (pyValue == nullptr) {
            if (mandatory) {
                TE_ERRLOG("OpTaskResult parse error, no %s present.", key.c_str());
                return nullptr;
            } else {
                continue;
            }
        }
        int rc = HandleManager::Instance()._PyArg_Parse(pyValue, pyType.c_str(), value);
        TE_FUSION_CHECK_WITH_DUMP_PYERR(rc == 0, {
            TE_ERRLOG("Parse opTask result with %s failed.", key.c_str());
            return nullptr;
        });
    }

    bool errMessage = (statusValue == 1 && typeValue != OP_TASK_TYPE_FUSION && pyErrMsgTuple != nullptr &&
                       PyTuple_Check(pyErrMsgTuple) != 0);
    if (errMessage) {
        PyErrMessageReport(pyErrMsgTuple);
    }

    std::string jsonFilePath;
    std::string pattern;
    std::string coreType;
    std::string compileInfo;
    std::string compileInfoKey;

    if (opRes != nullptr &&
        !ParseResult(opRes, typeValue, jsonFilePath, pattern, coreType,
                     compileInfo, compileInfoKey, fusionCheckResCode)) {
        TE_ERRLOG("Failed to parse result, opRes is %s, statusValue is %d, resValue is %s.",
                  opRes, statusValue, resValue);
        return nullptr;
    }

    if (fusionCheckResCode != CHECK_RES_INITIAL_VALUE) {
        statusValue = fusionCheckResCode;
    }
    OpBuildTaskResultPtr resultPtr = std::make_shared<OpBuildTaskResult>();

    resultPtr->type = typeValue;
    resultPtr->graphId = gidValue;
    resultPtr->taskId = tidValue;

    resultPtr->statusCode = statusValue;
    resultPtr->result = resValue ? resValue : "";
    resultPtr->infoMsg = msgValue ? msgValue : "";

    resultPtr->errArgs = errArgsValue ? errArgsValue : "";
    resultPtr->pyExceptMsg = pyExceptMsgValue ? pyExceptMsgValue : "";

    // built compile info
    resultPtr->compile_info_key = compileInfoKey;
    // built compile info
    resultPtr->compile_info_str = compileInfo;

    // json file path
    resultPtr->jsonFilePath = jsonFilePath;
    resultPtr->backToSingleOpBinReuse = false;

    resultPtr->preCompileRetType = PreCompileResultType::Online;
    TE_FUSION_MAKE_SHARED(resultPtr->preCompileRetPtr = std::make_shared<PreCompileResult>(pattern), return nullptr);
    resultPtr->preCompileRetPtr->coreType = coreType;
    resultPtr->preCompileRetPtr->prebuiltOptions = prebuiltOptions ? prebuiltOptions : "";

    resultPtr->compileRetType = CompileResultType::Online;

    TE_DBGLOG("Finish parsing compile result json file path[%s].", jsonFilePath.c_str());

    return resultPtr;
}

bool PythonAdapterManager::ParseResult(const char *opRes, int typeValue, std::string &jsonFilePath,
                                       std::string &pattern, std::string &coreType, std::string &compileInfo,
                                       std::string &compileInfoKey, int &fusionCheckResCode)
{
    if (opRes == nullptr) {
        TE_ERRLOG("opRes is empty!");
        return false;
    }

    nlohmann::json jsonOpRes;
    try {
        jsonOpRes = nlohmann::json::parse(opRes);
    } catch (std::exception &e) {
        TE_ERRLOG("Failed to parse json_str, the json_str is %s and the reason is %s", opRes, e.what());
        return false;
    }

    if (jsonOpRes.contains("fusion_check_result")) {
        fusionCheckResCode = jsonOpRes["fusion_check_result"].get<int>();
    }

    // prebuild task
    TE_DBGLOG("typeValue: %d.", typeValue);
    if (typeValue == 0) {
        if (jsonOpRes.contains("pattern")) {
            pattern = jsonOpRes["pattern"].get<std::string>();
            TE_DBGLOG("pattern: %s.", pattern.c_str());
        } else {
            TE_ERRLOG("There is no discernible pattern in the build result! jsonOpRes=%s.", jsonOpRes.dump().c_str());
            return false;
        }

        if (jsonOpRes.contains("core_type")) {
            coreType = jsonOpRes["core_type"].get<std::string>();
            TE_DBGLOG("coreType: %s.", coreType.c_str());
        } else {
            TE_DBGLOG("There is no core_type in the build result!");
        }
        return true;
    }
    // other type, such as single op build task 1 or fusion op build task 2 or autotune task 2.
    if (!jsonOpRes.contains("json_file_path")) {
        TE_WARNLOG("Failed to get json_file_path from jsonOpRes=%s.", jsonOpRes.dump().c_str());
        return true;
    }

    jsonFilePath = jsonOpRes["json_file_path"].get<std::string>();
    TE_DBGLOG("JsonFilePath is %s.", jsonFilePath.c_str());

    if (jsonFilePath.empty()) {
        TE_WARNLOG("json_file_path is empty, from jsonOpRes=%s.", jsonOpRes.dump().c_str());
        return false;
    }

    std::string realJsonFilePath = RealPath(jsonFilePath);
    if (!realJsonFilePath.empty()) {
        return GetCompileInfo(realJsonFilePath, compileInfo, compileInfoKey);
    }
    if (jsonOpRes.contains("compile_info")) {
        nlohmann::json jsonCompileInfo = jsonOpRes.at("compile_info");
        if (!jsonCompileInfo.is_null()) {
            compileInfo = jsonOpRes.at("compile_info").dump();
            if (!compileInfo.empty()) {
                bool res = PythonApiCall::Instance().GenerateStrSha256HashValue(compileInfo, compileInfoKey);
                TE_FUSION_CHECK(!res, {
                    TE_ERRLOG("Failed to generate the compile info key for compileInfo: %s.", compileInfo.c_str());
                    return false;
                });
            }
        }
    }
    return true;
}

bool PythonAdapterManager::GetCompileInfo(const std::string &jsonFilePath, std::string &compileInfo,
                                          std::string &compileInfoKey)
{
    nlohmann::json jsonInfo;
    if (!TeFileUtils::GetJsonValueFromJsonFile(jsonFilePath, jsonInfo)) {
        TE_WARNLOG("Failed to get jsonValue from jsonFile [%s].", jsonFilePath.c_str());
        return true;
    }

    if (jsonInfo.contains("compileInfo")) {
        compileInfo = jsonInfo.at("compileInfo").dump();
        if (!compileInfo.empty()) {
            bool res = PythonApiCall::Instance().GenerateStrSha256HashValue(compileInfo, compileInfoKey);
            TE_FUSION_CHECK(!res, {
                TE_ERRLOG("Failed to generate the compile info key for jsonFile: %s.",
                          jsonFilePath.c_str());
                return false;
            });
        }
    } else {
        TE_DBGLOG("There is no compileInfo in jsonFilePath %s.", jsonFilePath.c_str());
    }
    return true;
}

bool PythonAdapterManager::DispatchBinarySingleOpCompileTask(const std::string &taskDescStr) const
{
    PyLockGIL pyLockGIL;
    TE_DBGLOG("get PyLockGIL lock.");
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleCompileTaskMgr_, "dispatch_binary_single_op_compile_task", "s",
                                              taskDescStr.c_str());
    return pyRes != nullptr;
}
bool PythonAdapterManager::DispatchBinaryFusionOpCompileTask(const std::string &taskDescStr) const
{
    PyLockGIL pyLockGIL;
    TE_DBGLOG("get PyLockGIL lock.");
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleCompileTaskMgr_, "dispatch_binary_fusion_op_compile_task", "s",
                                              taskDescStr.c_str());
    return pyRes != nullptr;
}

/*
 * @brief: report python error message
 * @param [in] pyRes: PyObject value
 */
void PythonAdapterManager::PyErrMessageReport(PyObject *pyErrMsgTuple)
{
    std::vector<std::map<std::string, std::string>> mapListArgs;
    PyWrapper::UploadPyException(pyErrMsgTuple, mapListArgs);
    std::map<std::string, std::string>::iterator it;
    for (size_t i = 0; i < mapListArgs.size(); ++i) {
        std::map<std::string, std::string> mapArgs = mapListArgs[i];
        std::string errorCode = "";
        std::string errorType = "";
        std::string datailedCause = "";
        it = mapArgs.find("errCode");
        if (it != mapArgs.end()) {
            errorCode = it->second;
        }
        it = mapArgs.find("type");
        if (it != mapArgs.end()) {
            errorType = it->second;
        }
        it = mapArgs.find("detailed_cause");
        if (it != mapArgs.end()) {
            datailedCause = it->second;
        }
        if (!errorCode.empty()) {
            if (errorType == kInnerErrType) {
                TeInnerErrMessageReport(errorCode, datailedCause);
            } else {
                TeErrMessageReport(errorCode, mapArgs);
            }
        } else {
            TE_WARNLOG("ErrCode in py exception is empty.");
        }
    }
}
}
}