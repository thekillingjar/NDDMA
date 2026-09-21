/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "python_adapter/python_api_call.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_error_code.h"
#include "inc/te_fusion_util_constants.h"
#include "common/fusion_common.h"
#include "assemble_json/te_json_utils.h"
#include "python_adapter/pyobj_assemble_utils.h"
#include "graph/ge_context.h"
#include "common/tbe_op_info_cache.h"
#include "cache/te_cache_manager.h"

namespace te {
namespace fusion {
using namespace ge;
using namespace nlohmann;

std::mutex PythonApiCall::mtx_;

namespace {
PyObject* GenPyContextDict(const std::string &opFuncName, const std::string opModule, const std::string opType)
{
    PyObject *pyContextDict = HandleManager::Instance().TE_PyDict_New();
    if (pyContextDict == nullptr) {
        TE_ERRLOG("Failed to create py contextDict.");
        return nullptr;
    }
    int ret = 0;
    std::string virtualTypeStr;
    const std::string virtualType = "virtual_type";
    if (ge::GetContext().GetOption("ge.virtual_type", virtualTypeStr) == ge::GRAPH_SUCCESS && !virtualTypeStr.empty()) {
        PyObject *pyVirtualType = HandleManager::Instance()._Py_BuildValue("s", virtualTypeStr.c_str());
        AUTO_PY_DECREF(pyVirtualType);
        ret = HandleManager::Instance().TE_PyDict_SetItemString(pyContextDict, virtualType.c_str(), pyVirtualType);
        TE_FUSION_CHECK((ret != 0), {
            TE_ERRLOG("module [%s] opFuncName [%s] opType [%s], Building PyDict_SetItemString [%s] failed.",
                       opModule.c_str(), opFuncName.c_str(), opType.c_str(),  virtualType.c_str());
            return nullptr;
        });
    }
    // Transparently transmit the ge.jit_compile configuration value to Python
    std::string jitCompileStr;
    const std::string jitCompile = "jit_compiler";
    if (ge::GetContext().GetOption("ge.jit_compile", jitCompileStr) == ge::GRAPH_SUCCESS && !jitCompileStr.empty()) {
        PyObject *pyJitCompile = HandleManager::Instance()._Py_BuildValue("s", jitCompileStr.c_str());
        AUTO_PY_DECREF(pyJitCompile);
        ret = HandleManager::Instance().TE_PyDict_SetItemString(pyContextDict, jitCompile.c_str(), pyJitCompile);
        TE_FUSION_CHECK((ret != 0), {
            TE_ERRLOG("module [%s] opFuncName [%s], opType [%s], Build PyDict_SetItemString [%s] failed.",
                      opModule.c_str(), opFuncName.c_str(), opType.c_str(), jitCompile.c_str());
            return nullptr;
        });
    }
    return pyContextDict;
}
}

PythonApiCall::PythonApiCall() : l1SpaceSize_(-1) {}

PythonApiCall::~PythonApiCall() {}

PythonApiCall &PythonApiCall::Instance()
{
    static PythonApiCall pythonApiCallInstance;
    return pythonApiCallInstance;
}

void PythonApiCall::PythonApiCallInit(PyObjectPtr &pyModuleFusionMgr, PyObjectPtr &pyModuleFusionUtil,
                                      PyObjectPtr &pyModuleCompileTaskMgr, PyObjectPtr &pyModuleCcePolicy)
{
    pyModuleFusionMgr_ = pyModuleFusionMgr;
    pyModuleFusionUtil_ = pyModuleFusionUtil;
    pyModuleCompileTaskMgr_ = pyModuleCompileTaskMgr;
    pyModuleCcePolicy_ = pyModuleCcePolicy;
}

template<typename... Args>
bool PythonApiCall::CallPyFuncWithTbeOpInfo(const TbeOpInfo &tbeOpInfo, PyObjectPtr &pyRes, const bool &isSingleOpBuild,
                                            const std::string &pyFunc, const std::string &argFormat, Args... args)
{
    // convert op all parameter from class to PyObject
    std::string kernelName;
    PyObject *pyInputs = nullptr;
    PyObject *pyOutputs = nullptr;
    PyObject *pyAttrs = nullptr;
    bool res = AssembleOpArgs(tbeOpInfo, kernelName, pyInputs, pyOutputs, pyAttrs, isSingleOpBuild);
    TE_FUSION_CHECK(!res, {
        REPORT_TE_INNER_ERROR("Failed to assemble op args with pyFunc[%s].", pyFunc.c_str());
        return false;
    });

    AUTO_PY_DECREF(pyInputs);
    AUTO_PY_DECREF(pyOutputs);
    AUTO_PY_DECREF(pyAttrs);

    std::string format = "(OOO)" + argFormat;
    pyRes = PyWrapper::CallMethodNoErrPrint(pyModuleFusionMgr_, pyFunc.c_str(), format.c_str(),
                                            pyInputs, pyOutputs, pyAttrs, args...);
    PyObject *pyNone = HandleManager::Instance().get_py_none();
    TE_FUSION_CHECK_WITH_DUMP_PYERR((pyNone == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR,
            "Failed to convert parameter from none to PyObject.");
        return false;
    });

    TE_FUSION_CHECK((pyRes == nullptr || pyRes.get() == pyNone), {
        TE_INFOLOGF("Call python func[%s] not success, need to check op info: op inputs: %s, outputs: %s, attrs: %s.",
                    pyFunc.c_str(), PyObjectToStr(pyInputs).c_str(), PyObjectToStr(pyOutputs).c_str(),
                    PyObjectToStr(pyAttrs).c_str());
        return false;
    });
    TE_DBGLOGF("call python func[%s], op info: op inputs: %s, outputs: %s, attrs: %s.", pyFunc.c_str(),
        PyObjectToStr(pyInputs).c_str(), PyObjectToStr(pyOutputs).c_str(), PyObjectToStr(pyAttrs).c_str());
    return true;
}

PyObject *PythonApiCall::GetStrDictPyObj(const std::map<std::string, std::string> &infoMap) const
{
    PyObject *pyInfoDict = HandleManager::Instance().TE_PyDict_New();
    TE_FUSION_CHECK((pyInfoDict == nullptr), {
        TE_ERRLOG("Failed to create pyDict.");
        return nullptr;
    });
    bool res = AssemleStringDict(infoMap, pyInfoDict);
    TE_FUSION_CHECK(!res, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to assemble string dict.");
        AUTO_PY_DECREF(pyInfoDict);
        return nullptr;
    });
    return pyInfoDict;
}

PyObject *PythonApiCall::GetGeneralizeConfig(const TbeOpInfo &tbeOpInfo, const string &buildType) const
{
    std::map<std::string, std::string> generalizeConfig;
    if (buildType == FUZZILY_BUILD_TYPE) {
        generalizeConfig["mode"] = "keep_rank";
    } else if (buildType == BINARY_BUILD_TYPE) {
        generalizeConfig["mode"] = "all_shape";
    } else {
        TE_WARNLOG("Build type[%s] is invalid", buildType.c_str());
        return nullptr;
    }

    if (tbeOpInfo.IsSingleOpScene()) {
        generalizeConfig["single_op"] = "true";
    } else {
        generalizeConfig["single_op"] = "false";
    }

    return GetStrDictPyObj(generalizeConfig);
}

bool PythonApiCall::ParsePyResultOfString(const PyObjectPtr &pyRes, std::string &resString) const
{
    char *pstr = nullptr;
    int ires = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "s", &pstr);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires == 0), {
        TE_ERRLOG("Failed to parse function result string from Python object. Return code: [%d]", ires);
        return false;
    });

    TE_FUSION_CHECK((!pstr), {
        TE_ERRLOG("Failed to parse result");
        return false;
    });

    resString = pstr;
    return true;
}

bool PythonApiCall::ParsePyResultOfListString(const PyObjectPtr &pyRes, std::vector<std::string> &resList) const
{
    PyObject *pyResUniqueKeys = pyRes.get();
    Py_ssize_t listLen = HandleManager::Instance().TE_PyList_Size(pyResUniqueKeys);
    if (listLen < 0) {
        TE_ERRLOG("pyRes return is not a list.");
        return false;
    }
    if (listLen == 0) {
        TE_ERRLOG("pyRes returned an empty list.");
        return false;
    }
    TE_DBGLOG("Get list length is [%d].", listLen);
    for (ssize_t i = 0; i < listLen; ++i) {
        PyObject *pykey = nullptr;
        char *pyStr = nullptr;
        pykey = HandleManager::Instance().TE_PyList_GetItem(pyResUniqueKeys, i);
        TE_FUSION_CHECK_WITH_DUMP_PYERR(pykey == nullptr, {
            TE_ERRLOG("Failed to get unique pykey.");
            return false;
        });
        int ires = HandleManager::Instance()._PyArg_Parse(pykey, "s", &pyStr);
        TE_FUSION_CHECK_WITH_DUMP_PYERR((ires == 0), {
            TE_ERRLOG("Failed to parse function result string from Python object. Return code: [%d]", ires);
            return false;
        });
        resList.emplace_back(pyStr);
    }
    return true;
}

bool PythonApiCall::GetOpSpecificInfo(const TbeOpInfo &tbeOpInfo, std::string &opSpecificInfo)
{
    TE_DBGLOG("Start to get op specific info.");
    std::string opModule;
    (void)tbeOpInfo.GetModuleName(opModule);
    const std::string &opName = tbeOpInfo.GetName();

    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes;
    ERR_CODE res = CallOpFunc(tbeOpInfo, FUNC_GET_SPECIFIC_INFO, false, pyRes);
    TE_FUSION_CHECK(res == ERR_CODE::ERR_FAIL, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Node[%s] call get_op_specific_info failed.", opName.c_str());
        return false;
    });

    TE_FUSION_CHECK(!ParsePyResultOfString(pyRes, opSpecificInfo), {
        TE_ERRLOG("Failed to ParsePyResultOfString func[%s %s].", opModule.c_str(), FUNC_GET_SPECIFIC_INFO.c_str());
        return false;
    });

    TE_DBGLOGF("Op[%s] specific info is: %s", opModule.c_str(), opSpecificInfo.c_str());
    return true;
}

bool PythonApiCall::GetOpUniqueKeys(const TbeOpInfo &tbeOpInfo, const std::string &jsonStr,
                                    std::vector<std::string> &opUniqueKeyList)
{
    const std::string &opName = tbeOpInfo.GetName();
    const std::string &opType = tbeOpInfo.GetOpType();
    const std::string &opImplMode = tbeOpInfo.GetOpImplMode();

    PyLockGIL gil;
    PyObjectPtr pyRes;
    bool ret = CallPyFuncWithTbeOpInfo(tbeOpInfo, pyRes, true, FUNC_GENERATE_OP_UNIQUE_KEYS,
                                       "sss", opType.c_str(), opImplMode.c_str(), jsonStr.c_str());
    TE_FUSION_CHECK(!ret, {
        TE_DBGLOGF("Unable to call python func[%s].", FUNC_GENERATE_OP_UNIQUE_KEYS.c_str());
        return false;
    });

    TE_FUSION_CHECK(!ParsePyResultOfListString(pyRes, opUniqueKeyList), {
        TE_ERRLOG("Failed to parse result of func[%s].", FUNC_GENERATE_OP_UNIQUE_KEYS.c_str());
        return false;
    });

    TE_INFOLOG("Op unique key of [%s, %s] size is [%d].", opName.c_str(), opType.c_str(), opUniqueKeyList.size());
    for (auto &opUniqueKey : opUniqueKeyList) {
        TE_DBGLOG("Node of op[%s, %s] with opUniqueKey [%s].", opName.c_str(), opType.c_str(), opUniqueKey.c_str());
    }
    return true;
}

bool PythonApiCall::IsOpImplModeSupported(const ConstTbeOpInfoPtr &tbeOpInfoPtr, bool &isSupported)
{
    std::string opModule;
    std::string opFuncName;
    tbeOpInfoPtr->GetModuleName(opModule);
    tbeOpInfoPtr->GetFuncName(opFuncName);

    PyLockGIL pyLockGIL;
    if (!UpdateSingleOpModule(*tbeOpInfoPtr, opModule)) {
        TE_WARNLOG("Failed to update op module[%s].", opModule.c_str());
        return false;
    }

    PyObjectPtr pyRes = PyWrapper::CallMethodNoErrPrint(pyModuleFusionMgr_, "check_op_impl_mode", "ss",
                                                        opModule.c_str(), opFuncName.c_str());
    if (pyRes == nullptr) {
        TE_WARNLOG("Failed to call check_op_impl_mode, op module[%s], op func name[%s].",
                   opModule.c_str(), opFuncName.c_str());
        return false;
    }

    isSupported = HandleManager::Instance().TE_PyObject_IsTrue(pyRes.get());
    return true;
}

void PythonApiCall::ParseCheckResult(PyObject *resPtr, const std::string &opModule,
                                     CheckSupportedResult &isSupport, std::string &reason) const
{
    std::string result;
    bool isResTuple = PyTuple_Check(resPtr);
    if (!isResTuple) {
        result = PyObjectToStr(resPtr);
        reason = "Null";
    } else {
        PyObject *parsedResult = nullptr;
        char* parsedReason = nullptr;
        int res = HandleManager::Instance()._PyArg_ParseTuple(resPtr, "Os", &parsedResult, &parsedReason);
        if (res == 0) {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to parse return of function check_supported with reason.");
            result = "False";
            isSupport = NOT_SUPPORTED;
            reason = "Failed to Parse result from operator implementation";
            return;
        }
        result = PyObjectToStr(parsedResult);
        if (parsedReason != nullptr) {
            reason = parsedReason;
        } else {
            reason = "Null";
        }
    }

    if (result == "True") {
        isSupport = FULLY_SUPPORTED;
        TE_DBGLOG("opModule is %s, check support result is fully supported.", opModule.c_str());
    } else if (result == "False") {
        isSupport = NOT_SUPPORTED;
        TE_DBGLOG("opModule is %s, check support result is not supported.", opModule.c_str());
    } else if (result == "Unknown") {
        isSupport = PARTIALLY_SUPPORTED;
        TE_DBGLOG("opModule is %s, check support result is partially supported.", opModule.c_str());
    } else {
        isSupport = NOT_SUPPORTED;
        TE_DBGLOG("opModule is %s, check support result is %s, not supported.", opModule.c_str(), result.c_str());
    }

    if (!reason.empty() && isResTuple) {
        TE_DBGLOG("UnSupported reason is %s", reason.c_str());
    }
}

ERR_CODE PythonApiCall::CallOpFunc(const TbeOpInfo &opinfo, const std::string &opFuncName,
                                   const bool &isNeedJudgeOpFuncName, PyObjectPtr &pyRes)
{
    const std::string &opType = opinfo.GetOpType();
    const std::string &opImplMode = opinfo.GetOpImplMode();
    std::string opModule;
    (void)opinfo.GetModuleName(opModule);
    TE_FUSION_CHECK(!UpdateSingleOpModule(opinfo, opModule), {
        TE_WARNLOG("Failed to update the single op module for %s.", opModule.c_str());
        return ERR_CODE::ERR_FAIL;
    });

    if (isNeedJudgeOpFuncName) {
        TE_FUSION_CHECK(!CheckFuncInPyModule(opFuncName, opModule), {
            TE_WARNLOG("Failed to CheckFuncInPyModule func[%s %s].", opModule.c_str(), opFuncName.c_str());
            return ERR_CODE::ERR_FAIL;
        });
    }

    std::string softSyncStr = "False";
    ge::NodePtr nodePtr = opinfo.GetNode();
    if (nodePtr != nullptr) {
        bool softSyncOp = false;
        (void)ge::AttrUtils::GetBool(nodePtr->GetOpDesc(), SOFTSYNC_DYNAMIC_IMPL, softSyncOp);
        softSyncStr = softSyncOp ? "True" : "False";
    }
    PyObject *pyContextDict = GenPyContextDict(opFuncName, opModule, opType);
    TE_FUSION_CHECK(pyContextDict == nullptr, {
        TE_ERRLOG("Failed to generate context dict for %s.", opModule.c_str());
        return ERR_CODE::ERR_FAIL;
    });
    AUTO_PY_DECREF(pyContextDict);
    bool res = CallPyFuncWithTbeOpInfo(opinfo, pyRes, false, "call_op_func", "sssssO",
                                       opModule.c_str(), opFuncName.c_str(), opType.c_str(), opImplMode.c_str(),
                                       softSyncStr.c_str(), pyContextDict);
    if (!res) {
        return ERR_CODE::ERR_FAIL;
    }
    return ERR_CODE::NO_ERR;
}
/*
 * @brief: check tbe op capability
 * @param [in] opinfo: op total parameter set
 * @param [out] IsSupport: result to support or not
 * @return bool: check process ok or not
 */
bool PythonApiCall::CheckSupported(const TbeOpInfo &opinfo, CheckSupportedResult &isSupport, std::string &reason)
{
    std::string opFuncName = FUNC_CHECK_SUPPORTED;
    std::string opModule;
    (void)opinfo.GetModuleName(opModule);
    std::string checkName = opinfo.GetCheckSupportedFunc();
    if (checkName != "") {
        opFuncName = checkName;
    }

    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes;
    ERR_CODE res = CallOpFunc(opinfo, opFuncName, false, pyRes);
    TE_FUSION_CHECK(res == ERR_CODE::ERR_FAIL, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Node[%s] call check_supported failed.", opModule.c_str());
        return false;
    });

    PyObject *resPtr = pyRes.get();
    ParseCheckResult(resPtr, opModule, isSupport, reason);

    return true;
}

bool PythonApiCall::SelectTbeOpFormat(const TbeOpInfo &opinfo, std::string &opDtypeFormat)
{
    TE_DBGLOG("Fusion Manager starts to select tbe OP format");

    const std::string &opName = opinfo.GetName();
    std::string opModule;
    (void)opinfo.GetModuleName(opModule);

    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes;
    ERR_CODE res = CallOpFunc(opinfo, FUNC_OP_SELECT_FORMAT, false, pyRes);
    TE_FUSION_CHECK(res == ERR_CODE::ERR_FAIL, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Node[%s] call op_select_format failed.", opName.c_str());
        return false;
    });

    TE_FUSION_CHECK(!ParsePyResultOfString(pyRes, opDtypeFormat), {
        TE_ERRLOG("Failed to ParsePyResultOfString func[%s %s].", opModule.c_str(), FUNC_OP_SELECT_FORMAT.c_str());
        return false;
    });

    TE_DBGLOGF("Select tbe op format[%s] result: %s", opModule.c_str(), opDtypeFormat.c_str());
    return true;
}

LX_QUERY_STATUS PythonApiCall::GetOpInfo(const TbeOpInfo &tbeOpInfo, std::string &result)
{

    TE_DBGLOG("Fusion manager start to query LxFusion info");
    std::string opModule;
    (void)tbeOpInfo.GetModuleName(opModule);

    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes;
    ERR_CODE res = CallOpFunc(tbeOpInfo, FUNC_GET_OP_SUPPORT_INFO, true, pyRes);
    TE_FUSION_CHECK(res == ERR_CODE::ERR_FAIL, {
        TE_WARNLOG("Failed to call op func %s, need to check op info: module name[%s].",
                   FUNC_GET_OP_SUPPORT_INFO.c_str(), opModule.c_str());
        return LX_QUERY_FAIL;
    });

    char *pstr = nullptr;
    int ires = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "s", &pstr);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires == 0), {
        TE_ERRLOG("Failed to parse function result string in python object. Module[%s], Func[%s]. ires:[%d]",
                  opModule.c_str(), FUNC_GET_OP_SUPPORT_INFO.c_str(), ires);
        return LX_QUERY_FAIL;
    });

    TE_FUSION_CHECK((!pstr), {
        TE_ERRLOG("Failed to parse func[%s %s] result", opModule.c_str(), FUNC_GET_OP_SUPPORT_INFO.c_str());
        return LX_QUERY_FAIL;
    });

    result = pstr;
    TE_DBGLOG("[%s %s] queries LxFusion info success, and result: %s",
              opModule.c_str(), FUNC_GET_OP_SUPPORT_INFO.c_str(), pstr);
    return LX_QUERY_SUCC;
}

bool PythonApiCall::CheckIsTbeGeneralizeFuncRegistered(const TbeOpInfo &tbeOpInfo, bool &hasRegisteredFunc)
{
    const std::string &opType = tbeOpInfo.GetOpType();
    std::string opModule;
    (void)tbeOpInfo.GetModuleName(opModule);
    PyLockGIL gil;
    TE_FUSION_CHECK(!UpdateSingleOpModule(tbeOpInfo, opModule), {
        TE_ERRLOG("Failed to update and import single op module for %s.", opType.c_str());
        return false;
    });

    TE_DBGLOG("node[%s] CheckIsTbeGeneralizeFuncRegistered for opModule[%s].", opType.c_str(), opModule.c_str());
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleFusionMgr_, "is_generalize_func_register_from_c", "ss",
                                              opType.c_str(), opModule.c_str());
    TE_FUSION_CHECK((pyRes == nullptr), {
        TE_ERRLOGF("Failed to call op func is_generalize_func_register_from_c with opType[%s] and opModule[%s].",
                   opType.c_str(), opModule.c_str());
        return false;
    });

    int parsedResult = 0;
    int32_t ires = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "i", &parsedResult);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires == 0), {
        TE_ERRLOG("is_generalize_func_register_from_c return null, op_type: %s", opType.c_str());
        return false;
    });

    if (parsedResult == 1) {
        hasRegisteredFunc = true;
    } else {
        hasRegisteredFunc = false;
    }
    TE_DBGLOG("node[%s] %s register generalize func.", opType.c_str(), hasRegisteredFunc ? "" : "not");
    return true;
}

bool PythonApiCall::GetCheckAndGeneralizedResFromTbe(const TbeOpInfo &tbeOpInfo,  const string &buildType,
                                                     nlohmann::json &generalizedRes)
{
    const std::string &opName = tbeOpInfo.GetName();
    const std::string &opType = tbeOpInfo.GetOpType();
    const std::string &opImplMode = tbeOpInfo.GetOpImplMode();
    std::string opModule;
    (void)tbeOpInfo.GetModuleName(opModule);

    PyLockGIL pyLockGIL;
    std::string *pyModulePath = nullptr;
    bool res = NormalizeModuleName(opModule, pyModulePath);
    TE_FUSION_CHECK(!res, {
        REPORT_TE_INNER_ERROR("Failed to normalize module name [%s] for node [%s].", opModule.c_str(), opName.c_str());
        return false;
    });
    std::string opsPathNamePrefix = tbeOpInfo.GetOpsPathNamePrefix();
    size_t pos = opModule.find_last_of('.');
    if (pos != string::npos) {
        te::fusion::UpdateStaticOpModulePath(opsPathNamePrefix, opModule, pos, false);
    }

    PyObject *pyGeneralizeConfig = GetGeneralizeConfig(tbeOpInfo, buildType);
    TE_FUSION_CHECK((pyGeneralizeConfig == nullptr), {
        REPORT_TE_INNER_ERROR("Failed to convert generalize config to py obj for node[%s].", opName.c_str());
        return false;
    });
    AUTO_PY_DECREF(pyGeneralizeConfig);

    PyObjectPtr pyRes;
    res = CallPyFuncWithTbeOpInfo(tbeOpInfo, pyRes, false, "generalize_shape_and_range_from_c", "sssO",
                                  opType.c_str(), opModule.c_str(), opImplMode.c_str(), pyGeneralizeConfig);
    TE_FUSION_CHECK(!res, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Node[%s] CallPyFuncWithTbeOpInfo failed.", opName.c_str());
        return false;
    });

    char* pJson = nullptr;
    int32_t ires = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "s", &pJson);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires == 0 || pJson == nullptr), {
        REPORT_TE_INNER_ERROR("Func generalize_shape_and_range_from_c of node[%s] return null.", opName.c_str());
        return false;
    });

    try {
        generalizedRes = json::parse(pJson);
    } catch (std::exception &e) {
        REPORT_TE_INNER_ERROR("Failed to parse json_str, the json_str is %s and the failure reason is %s", pJson, e.what());
        return false;
    }
    TE_DBGLOGF("Generalize result from tbe of node[%s] is: %s.", opName.c_str(), generalizedRes.dump().c_str());
    return true;
}

bool PythonApiCall::ParseDynamicShapeRangeCheckFromTbe(nlohmann::json &generalizedRes, bool &isSupported,
                                                       std::vector<size_t> &upperLimitedInputIndexs,
                                                       std::vector<size_t> &lowerLimitedInputIndexs) const
{
    try {
        if (generalizedRes.find("result") == generalizedRes.end()) {
            return true;
        }

        const std::string checkRes = generalizedRes.at("result").get<std::string>();
        if (checkRes != UNSUPPORTED) {
            REPORT_TE_INNER_ERROR("Check result [%s] from tbe is invalid.", checkRes.c_str());
            return false;
        }

        isSupported = false;
        nlohmann::json reasonJson = generalizedRes["reason"];
        std::vector<size_t> unsupported_indexs = reasonJson.at("param_index").get<std::vector<size_t>>();
        std::vector<std::string> unsupported_type = reasonJson.at("type").get<std::vector<std::string>>();
        if (std::find(unsupported_type.begin(), unsupported_type.end(), RANGE_UPPER_LIMIT) == unsupported_type.end()) {
            lowerLimitedInputIndexs = unsupported_indexs;
            return true;
        }

        if (std::find(unsupported_type.begin(), unsupported_type.end(), RANGE_LOWER_LIMIT) == unsupported_type.end()) {
            upperLimitedInputIndexs = unsupported_indexs;
            return true;
        }

        if (unsupported_type.size() != unsupported_indexs.size()) {
            TE_WARNLOG("unsupported_type size[%d] not equal unsupported_indexs size[%d].",
                       unsupported_type.size(), unsupported_indexs.size());
            return false;
        }
        for (size_t i = 0; i < unsupported_type.size(); ++i) {
            if (unsupported_type[i] == RANGE_UPPER_LIMIT) {
                upperLimitedInputIndexs.push_back(unsupported_indexs[i]);
            } else {
                lowerLimitedInputIndexs.push_back(unsupported_indexs[i]);
            }
        }
    } catch (std::exception &e) {
        REPORT_TE_INNER_ERROR("Parsing checkRes from tbe failed, reason is: %s", e.what());
        return false;
    }
    return true;
}

bool PythonApiCall::DynamicShapeRangeCheck(const TbeOpInfo &tbeOpInfo, bool &isSupported,
                                           std::vector<size_t> &upperLimitedInputIndexs,
                                           std::vector<size_t> &lowerLimitedInputIndexs)
{
    const std::string &opName = tbeOpInfo.GetName();
    nlohmann::json generalizedRes;
    isSupported = true;
    if (!GetCheckAndGeneralizedResFromTbe(tbeOpInfo, FUZZILY_BUILD_TYPE, generalizedRes)) {
        TE_ERRLOG("GetCheckAndGeneralizedResFromTbe for node %s failed.", opName.c_str());
        return false;
    }

    for (auto singleInfo = generalizedRes.begin(); singleInfo != generalizedRes.end(); ++singleInfo) {
        json singleInfoJson = *singleInfo;
        if (!ParseDynamicShapeRangeCheckFromTbe(singleInfoJson, isSupported, upperLimitedInputIndexs,
                                                lowerLimitedInputIndexs)) {
            TE_ERRLOG("ParseDynamicShapeRangeCheckFromTbe for node %s failed.", opName.c_str());
            return false;
        }
    }

    return true;
}

// Todo: this functon is not needed, will be removed later
bool PythonApiCall::ImportOpFilePath(std::string &pathName)
{
    std::lock_guard<std::mutex> lock_guard(pySysPathMutex_);
    if (pySysPath_.find(pathName) == pySysPath_.end()) {
        // import file path
        PyLockGIL gil;
        HandleManager::Instance().TE_PyRun_SimpleString("import sys");
        std::string pathAppendCmd = "if '" + pathName + "' not in sys.path:" + "sys.path.append('" + pathName + "')";
        HandleManager::Instance().TE_PyRun_SimpleString(pathAppendCmd.c_str());
        PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleCompileTaskMgr_, "sync_syspath", "s", pathName.c_str());
        TE_FUSION_CHECK((pyRes == nullptr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to sync syspath %s", pathName.c_str());
            return false;
        });
        TE_INFOLOGF("Succeeded to sync syspath %s", pathName.c_str());
        (void)pySysPath_.emplace(pathName);
    }

    return true;
}

bool PythonApiCall::GetAttrInNode(const TbeOpInfoPtr &pOpInfo, const std::string &jsonStr, std::string &opArgs)
{
    TE_FUSION_NOTNULL(pOpInfo);

    PyLockGIL pyLockGIL;
    TbeOpInfo &opInfo = *pOpInfo;
    std::string kernelName;
    opInfo.SetKernelName(kernelName);
    PyObject *pyAttrs = nullptr;
    bool res = AssembleAttrs(opInfo, kernelName, pyAttrs, true);
    TE_FUSION_CHECK(!res, {
        TE_ERRLOGF("Failed to assemble op args.");
        return false;
    });
    AUTO_PY_DECREF(pyAttrs);

    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleFusionUtil_, "get_op_args_json", "sO", jsonStr.c_str(), pyAttrs);
    if (pyRes == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to parse return of function set_op_args_json.");
        return false;
    }
    char *opArgsStr = nullptr;
    int32_t ires = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "s", &opArgsStr);
    TE_INFOLOGF("opArgsStr is %s.", opArgsStr);
    if (ires == 0 || opArgsStr == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "get_op_args_json return null.");
        return false;
    }
    opArgs = opArgsStr;
    TE_DBGLOGF("JsonStr: %s, attrs: %s.", jsonStr.c_str(), PyObjectToStr(pyAttrs).c_str());
    return true;
}

bool PythonApiCall::GenerateStrSha256HashValue(const std::string &str, std::string &result) const
{
    if (str.empty()) {
        result = "";
        return true;
    }
    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleFusionMgr_, "get_str_sha256_hash_from_c", "s", str.c_str());
    if (pyRes == nullptr) {
        TE_WARNLOG("Failed to call get_str_sha256_hash_from_c, str=[%s].", str.c_str());
        return false;
    };

    char *pstr = nullptr;
    int32_t ires = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "s", &pstr);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires == 0 || pstr == nullptr), {
        TE_WARNLOG("Failed to parse the python result.");
        return false;
    });

    result = pstr;
    return true;
}

bool PythonApiCall::GetBinFileSha256Value(const char *binData, const size_t binSize, std::string &sha256Val) const
{
    if (binData == nullptr || binSize == 0) {
        return true;
    }
    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes = PyWrapper::CallMethodNoErrPrint(pyModuleFusionMgr_, "get_binfile_sha256_hash_from_c",
                                              "y#", binData, binSize);
    if (pyRes == nullptr) {
        TE_WARNLOG("Failed to call get_str_sha256_hash_from_c.");
        return false;
    };

    char *pstr = nullptr;
    int32_t ires = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "s", &pstr);
    TE_FUSION_CHECK_WITH_DUMP_PYERR((ires == 0 || pstr == nullptr), {
        TE_WARNLOG("Failed to parse the python result.");
        return false;
    });

    sha256Val = pstr;
    return true;
}

/*
 * @brief: normalize module name
 * @param [in] ModuleName: original module name
 * @return bool: normaliable result
 */
bool PythonApiCall::NormalizeModuleName(std::string &ModuleName, std::string *pyModulePath)
{
    std::string localName;
    size_t pos;
    size_t lastLastPos = 0;
    size_t lastPos = 0;

    TE_FUSION_CHECK((ModuleName.empty()), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "OpModule should not be empty.");
        return false;
    });
    localName = ModuleName;
    pos = localName.find("/");
    while (pos != std::string::npos) {
        localName.replace(pos, 1, ".");
        lastLastPos = lastPos;
        lastPos = pos;
        pos = localName.find("/");
    }

    // deal with absolute module name
    if (ModuleName.at(0) == '/') {
        std::string pathName = ModuleName.substr(0, lastLastPos);
        localName = ModuleName.substr(lastLastPos + 1, ModuleName.length() - lastLastPos);

        pos = localName.find("/");
        while (pos != std::string::npos) {
            localName.replace(pos, 1, ".");
            pos = localName.find("/");
        }

        if (lastLastPos == 0) {
            pathName = "/";
        }

        if (pyModulePath != nullptr) {
            *pyModulePath = pathName;
        }

        TE_FUSION_CHECK(!ImportOpFilePath(pathName), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to import path %s.", pathName.c_str());
            return false;
        });
    }
    // return relative module name
    ModuleName = localName;

    return true;
}

bool PythonApiCall::UpdateSingleOpModule(const TbeOpInfo &tbeOpInfo, std::string &opModule, std::string *pyModulePath)
{
    // convert module name from custom to inner
    bool res = NormalizeModuleName(opModule, pyModulePath);
    TE_FUSION_CHECK(!res, {
        REPORT_TE_INNER_ERROR("Failed to normalize module name [%s] of node [%s].",
                              opModule.c_str(), tbeOpInfo.GetName().c_str());
        return false;
    });

    UpdateOpModuleFromStaicToDynamicAndAddPrefix(tbeOpInfo, opModule);

    TE_DBGLOG("Op[%s] new opModule is %s.", tbeOpInfo.GetName().c_str(), opModule.c_str());
    return true;
}

bool PythonApiCall::CheckFuncInPyModule(const std::string &opFuncName, const std::string &opModule) const
{
    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleFusionMgr_, "is_func_in_opModule_from_c", "ss",
                                              opFuncName.c_str(), opModule.c_str());
    if (pyRes == nullptr) {
        TE_WARNLOG("Failed to call is_func_in_opModule_from_c with %s", opModule.c_str());
        return false;
    }

    if (HandleManager::Instance().TE_PyObject_IsTrue(pyRes.get()) != 0) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "Python call succeessfully!");
        return true;
    }

    return false;
}

int64_t PythonApiCall::GetL1SpaceSize() const
{
    return l1SpaceSize_;
}

/*
 * @brief: config op L1 info to python
 * @param [in] opinfo: op info, include op name, module name, parameters, and so on
 * @return bool: save L1 info parameter to python ok or not
 */
bool PythonApiCall::SetL1SpaceSize(const int64_t l1SpaceValue)
{
    PyLockGIL pyLockGIL;
    const char *func = "set_L1_info";
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleCcePolicy_, func, "sl", "op_L1_space", l1SpaceValue);
    TE_FUSION_CHECK(pyRes == nullptr || HandleManager::Instance().TE_PyObject_IsTrue(pyRes.get()) == 0, {
        REPORT_TE_INNER_ERROR("Failed to call function [%s]. l1_size: %ld", func, l1SpaceValue);
        return false;
    });

    l1SpaceSize_ = l1SpaceValue;
    TE_DBGLOG("Set op params L1 info successfully. l1_size: %ld", l1SpaceValue);
    return true;
}

void PythonApiCall::ResetL1SpaceSize()
{
    if (l1SpaceSize_ == -1) {
        return;
    }
    PyLockGIL pyLockGIL;
    const char *func = "set_L1_info";
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleCcePolicy_, func, "si", "op_L1_space", -1);
    TE_FUSION_CHECK(pyRes == nullptr || HandleManager::Instance().TE_PyObject_IsTrue(pyRes.get()) == 0, {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING,
                           "Unable to call function [%s].", func);
        return;
    });

    l1SpaceSize_ = -1;
    TE_DBGLOG("L1Size reset");
}

/*
 * @brief: config op common cfg, like, ddk version, module and opname
 * @param [in] opinfo: op info, include op name, module name, parameters, and so on
 * @return bool: save compute parameter to python ok or not
 */
bool PythonApiCall::SetOpParams(const TbeOpInfo &opInfo, PyObject *pyOutputs)
{
    std::string opModule;
    (void)opInfo.GetModuleName(opModule);
    std::string opFuncName;
    opInfo.GetFuncName(opFuncName);

    // func begin log
    TE_DBGLOG("Start to set op params: Name=[%s], Module=[%s], FuncName=[%s].",
              opInfo.GetName().c_str(), opModule.c_str(), opFuncName.c_str());

    auto currentNode = opInfo.GetNode();
    std::string keyName = "";
    bool bres = TbeOpInfoCache::Instance().GetOpKeyNameByNode(currentNode.get(), keyName);
    TE_FUSION_CHECK(!bres, {
        TE_ERRLOG("Failed to get key[%s]", opInfo.GetName().c_str());
        return false;
    });

    PyLockGIL pyLockGIL;
    // convert op all parameter from class to PyObject
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleFusionMgr_, "op_params_to_json", "sO", keyName.c_str(), pyOutputs);
    TE_FUSION_CHECK((pyRes == nullptr), {
        TE_ERRLOG("Failed to call op_params_to_json, nodeName: %s", keyName.c_str());
        return false;
    });

    char *opParam = nullptr;
    int32_t ires = HandleManager::Instance()._PyArg_Parse(pyRes.get(), "s", &opParam);
    if (ires == 0 || opParam == nullptr) {
        REPORT_TE_INNER_ERROR("op_params_to_json return null, nodeName: %s", keyName.c_str());
        return false;
    }
    std::string opParamStr = opParam;
    TE_DBGLOG("SetOpParams name[%s], args[%s].", keyName.c_str(), opParamStr.c_str());
    TeCacheManager::Instance().SetOpArgsCache(keyName, opParamStr);

    if (!SetL1SpaceSize(opInfo.GetL1Space())) {
        TE_ERRLOG("Failed to set L1 info.");
        return false;
    }

    TE_DBGLOG("Node[%s] set op params successfully.", opInfo.GetName().c_str());
    return true;
}

bool PythonApiCall::DumpFusionJson(const std::string &jsonStr, const std::string &dumpPath) const
{
    if (jsonStr.empty() || dumpPath.empty()) {
        TE_WARNLOG("Json or dump path is empty.");
        return false;
    }
    PyLockGIL pyLockGIL;
    PyObjectPtr pyRes = PyWrapper::CallMethodNoErrPrint(pyModuleFusionUtil_, "dump_fusion_json", "ss",
                                                        jsonStr.c_str(), dumpPath.c_str());
    return pyRes != nullptr;
}

bool PythonApiCall::SyncOpTuneParams() const
{
    PyLockGIL pyLockGIL;
    PyObject *pyFalse = HandleManager::Instance().get_py_false();
    if (pyFalse == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get or pyFalse from class HandleManager.");
        return false;
    }
    std::string funcParams;
    PyObjectPtr pyRes = PyWrapper::CallMethod(pyModuleCompileTaskMgr_, "sync_op_tune_params", "ssOs",
                                              "tbe.common.tiling.tiling_api", "reset_repository", pyFalse,
                                              funcParams.c_str());
    if (pyRes == nullptr) {
        TE_ERRLOG("Failed to sync data [tbe.common.tiling.tiling_api][reset_repository].");
        return false;
    }
    return true;
}
} // namespace fusion
} // namespace te
