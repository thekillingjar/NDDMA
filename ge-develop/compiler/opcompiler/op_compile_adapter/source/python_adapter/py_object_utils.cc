/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "python_adapter/py_object_utils.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/te_config_info.h"
#include "common/te_context_utils.h"
#include "assemble_json/te_json_assemble.h"
#include "python_adapter/py_wrapper.h"
#include "python_adapter/py_decouple.h"
#include "ge_common/ge_api_types.h"

namespace te {
namespace fusion {
PyObject* PyObjectUtils::GenPySocInfo()
{
    TE_DBGLOG("Gen soc info, soc version[%s], core type[%s], core num[%s], l1 fusion[%s], l2 mode[%s], l2 fusion[%s].",
              TeConfigInfo::Instance().GetSocVersion().c_str(), TeConfigInfo::Instance().GetCoreType().c_str(),
              TeConfigInfo::Instance().GetAiCoreNum().c_str(), TeConfigInfo::Instance().GetL1Fusion().c_str(),
              TeConfigInfo::Instance().GetL2Mode().c_str(), TeConfigInfo::Instance().GetL2Fusion().c_str());
    PyObject *pySocInfoList = HandleManager::Instance()._Py_BuildValue("ssssss",
        TeConfigInfo::Instance().GetSocVersion().c_str(), TeConfigInfo::Instance().GetCoreType().c_str(),
        TeConfigInfo::Instance().GetAiCoreNum().c_str(), TeConfigInfo::Instance().GetL1Fusion().c_str(),
        TeConfigInfo::Instance().GetL2Mode().c_str(), TeConfigInfo::Instance().GetL2Fusion().c_str());
    TE_FUSION_CHECK(pySocInfoList == nullptr, {
        TE_ERRLOG("Failed to build py value with arguments: [%s, %s, %s, %s, %s, %s].",
                  TeConfigInfo::Instance().GetSocVersion().c_str(), TeConfigInfo::Instance().GetCoreType().c_str(),
                  TeConfigInfo::Instance().GetAiCoreNum().c_str(), TeConfigInfo::Instance().GetL1Fusion().c_str(),
                  TeConfigInfo::Instance().GetL2Mode().c_str(), TeConfigInfo::Instance().GetL2Fusion().c_str());
        return nullptr;
    });

    return pySocInfoList;
}

PyObject *PyObjectUtils::GenPyOptionDict()
{
    PyObject *opDebugLevel = HandleManager::Instance()._Py_BuildValue("s",
        TeConfigInfo::Instance().GetOpDebugLevelStr().c_str());
    PyObject *opDebugDir = HandleManager::Instance()._Py_BuildValue("s",
        TeConfigInfo::Instance().GetKernelMetaParentDir().c_str());
    PyObject *mdlBankPath = HandleManager::Instance()._Py_BuildValue("s",
        TeConfigInfo::Instance().GetMdlBankPath().c_str());
    PyObject *opBankPath = HandleManager::Instance()._Py_BuildValue("s",
        TeConfigInfo::Instance().GetOpBankPath().c_str());
    PyObject *kernelMetaTempDir = HandleManager::Instance()._Py_BuildValue("s",
        TeConfigInfo::Instance().GetKernelMetaTempDir().c_str());
    PyObject *vectorFpCeiling = HandleManager::Instance()._Py_BuildValue("s",
        TeConfigInfo::Instance().GetFpCeilingMode().c_str());
    PyObject *deviceId = HandleManager::Instance()._Py_BuildValue("s", TeConfigInfo::Instance().GetDeviceId().c_str());
    TE_FUSION_CHECK((opDebugLevel == nullptr || vectorFpCeiling == nullptr || mdlBankPath == nullptr ||
                     opBankPath == nullptr || deviceId == nullptr), {
                        TE_ERRLOG("Build PyObject failed with params: opDebugLevel: %s, vectorFpCeiling: %s, mdlBankPath: %s, opBankPath: %s",
                                  TeConfigInfo::Instance().GetSocVersion().c_str(), TeConfigInfo::Instance().GetOpDebugLevelStr().c_str(),
                                  TeConfigInfo::Instance().GetFpCeilingMode().c_str(), TeConfigInfo::Instance().GetMdlBankPath().c_str(),
                                  TeConfigInfo::Instance().GetOpBankPath().c_str());
                        return nullptr;
                    });

    AUTO_PY_DECREF(opDebugLevel);
    AUTO_PY_DECREF(opDebugDir);
    AUTO_PY_DECREF(mdlBankPath);
    AUTO_PY_DECREF(opBankPath);
    AUTO_PY_DECREF(kernelMetaTempDir);
    AUTO_PY_DECREF(vectorFpCeiling);
    AUTO_PY_DECREF(deviceId);
    PyObject *pySocParamDict = HandleManager::Instance().TE_PyDict_New();
    if (pySocParamDict == nullptr) {
        return nullptr;
    }
    TE_FUSION_CHECK((
         HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict, "op_debug_level", opDebugLevel) != 0 ||
         HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict, "op_debug_dir", opDebugDir) != 0 ||
         HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict, "vector_fp_ceiling", vectorFpCeiling) != 0 ||
         HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict, "mdl_bank_path", mdlBankPath) != 0 ||
         HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict, "op_bank_path", opBankPath) != 0 ||
         HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict,
                                                           "kernel_meta_temp_dir", kernelMetaTempDir) != 0 ||
         HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict, "device_id", deviceId) != 0), {
            TE_PY_DECREF(pySocParamDict);
            TE_ERRLOG("PyDict_SetItemString failed with options: [%s, %s, %s, %s, %s, %s].",
                      TeConfigInfo::Instance().GetOpDebugLevelStr().c_str(),
                      TeConfigInfo::Instance().GetKernelMetaParentDir().c_str(),
                      TeConfigInfo::Instance().GetFpCeilingMode().c_str(),
                      TeConfigInfo::Instance().GetMdlBankPath().c_str(),
                      TeConfigInfo::Instance().GetOpBankPath().c_str(),
                      TeConfigInfo::Instance().GetKernelMetaTempDir().c_str());
            return nullptr;
        });

    std::map<std::string, std::string> hardwareInfoMap;
    TeConfigInfo::Instance().GetHardwareInfoMap(hardwareInfoMap);
    for (const std::pair<const std::string, std::string> &hardwarePair : hardwareInfoMap) {
        if (hardwarePair.second.empty()) {
            continue;
        }
        PyObject *pyHardwareValue = HandleManager::Instance()._Py_BuildValue("s", hardwarePair.second.c_str());
        TE_DBGLOG("Build hardware py obj: [%s, %s].", hardwarePair.first.c_str(), hardwarePair.second.c_str());
        TE_FUSION_CHECK((pyHardwareValue == nullptr), {
            TE_PY_DECREF(pySocParamDict);
            TE_ERRLOG("Failed to build hardware object [%s, %s].", hardwarePair.first.c_str(), hardwarePair.second.c_str());
            return nullptr;
        });
        AUTO_PY_DECREF(pyHardwareValue);
        auto ret = HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict, hardwarePair.first.c_str(),
                                                                     pyHardwareValue);
        TE_FUSION_CHECK((ret != 0), {
            TE_PY_DECREF(pySocParamDict);
            TE_ERRLOG("Failed to set dictionary item [%s, %s]", hardwarePair.first.c_str(), hardwarePair.second.c_str());
            return nullptr;
        });
    }

    return pySocParamDict;
}

PyObject *PyObjectUtils::GenPyOptionsInfo(const std::vector<ConstTbeOpInfoPtr> &tbeOpInfoVec)
{
    std::map<std::string, std::string> optionsValues;
    if (!TeJsonAssemble::GenerateOptionsMap(tbeOpInfoVec, optionsValues)) {
        TE_WARNLOG("Failed to generate options map.");
        return HandleManager::Instance().TE_PyDict_New();
    }
    // enable vector core is only for online compile
    if (tbeOpInfoVec.size() == 1) {
        ConstTbeOpInfoPtr firstTbeOpInfo = tbeOpInfoVec.at(0);
        if (firstTbeOpInfo->GetVectorCoreType() == VectorCoreType::ENABLE && !firstTbeOpInfo->IsOriSingleOpScene()) {
            optionsValues.emplace(kEnableVectorCore, STR_TRUE);
        }
    }

    PyObject *pySocParamDict = HandleManager::Instance().TE_PyDict_New();
    if (pySocParamDict == nullptr) {
        TE_ERRLOG("pySocParamDict is null");
        return nullptr;
    }
    for (auto &option : optionsValues) {
        if (!GenPyOptionPair(option.first, option.second, pySocParamDict)) {
            TE_PY_DECREF(pySocParamDict);
            TE_ERRLOG("GenPyOptionPair[%s] failed with [%s]", option.first.c_str(), option.second.c_str());
            return nullptr;
        }
    }
    return pySocParamDict;
}

bool PyObjectUtils::GenPyOptionPair(const std::string &key, const std::string &val, PyObject *pySocParamDict) {
    if (val.empty()) {
        return true;
    }
    PyObject *optionsValue = HandleManager::Instance()._Py_BuildValue("s", val.c_str());
    TE_INFOLOG("Build options[%s] value is [%s].", key.c_str(), val.c_str());
    TE_FUSION_CHECK((optionsValue == nullptr), {
        TE_ERRLOG("build optionsValue[%s] failed with [%s]", key.c_str(), val.c_str());
        return false;
    });

    AUTO_PY_DECREF(optionsValue);
    auto ret = HandleManager::Instance().TE_PyDict_SetItemString(pySocParamDict, key.c_str(),
                                                                 optionsValue);
    TE_FUSION_CHECK((ret != 0), {
        TE_ERRLOG("Build PyDict_SetItemString[%s] failed with error: [%s]", key.c_str(), val.c_str());
        return false;
    });
    return true;
}

PyObject *PyObjectUtils::GenSuperKernelOptionsInfo(const std::map<std::string, std::string> &skOptions) {
    PyObject *pySocParamDict = HandleManager::Instance().TE_PyDict_New();
    if (pySocParamDict == nullptr) {
        TE_ERRLOG("pySocParamDict is null");
        return nullptr;
    }

    for (auto &option : skOptions) {
        if (!GenPyOptionPair(option.first, option.second, pySocParamDict)) {
            TE_PY_DECREF(pySocParamDict);
            TE_ERRLOG("GenSuperKernelOptionsInfo with key[%s] and val[%s] failed.",
                      option.first.c_str(), option.second.c_str());
            return nullptr;
        }
        TE_INFOLOG("GenSuperKernelOptionsInfo for super kernel with key[%s] and val[%s] success.",
                   option.first.c_str(), option.second.c_str());
    }
    TE_INFOLOG("GenSuperKernelOptionsInfo for the super kernel succeeded, the kernel_meta directory is [%s].",
               TeConfigInfo::Instance().GetKernelMetaParentDir().c_str());
    return pySocParamDict;
}
}
}
