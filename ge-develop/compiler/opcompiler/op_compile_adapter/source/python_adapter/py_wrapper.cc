/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "python_adapter/py_wrapper.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_error_code.h"
#include "common/common_utils.h"

namespace te {
namespace fusion {
constexpr int PY_EXCEPT_TUPLE_LEN = 2;

PyObject *PyWrapper::PyPrintExceptionFunc = nullptr;
PyObject *PyWrapper::PyUploadExceptionFunc = nullptr;

bool PyWrapper::InitPyLogger()
{
    if (PyPrintExceptionFunc != nullptr and PyUploadExceptionFunc != nullptr) {
        return true;
    }

    PyObjectPtr pyModule = ImportModule("te_fusion.log_util");
    if (pyModule == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to import te_fusion.log_util");
        return false;
    }
    PyObject *pyPrintFunc = HandleManager::Instance().TE_PyObject_GetAttrString(pyModule.get(), "get_py_exception_str");
    PyObject *pyUploadFunc =
        HandleManager::Instance().TE_PyObject_GetAttrString(pyModule.get(), "get_py_exception_list");
    if (pyPrintFunc == nullptr || pyUploadFunc == nullptr) {
        REPORT_TE_INNER_ERROR("PrintFunc or UploadFunc is nullptr.");
        return false;
    }
    if (PyPrintExceptionFunc) {
        TE_PY_DECREF(PyPrintExceptionFunc);
    }
    PyPrintExceptionFunc = pyPrintFunc;
    if (PyUploadExceptionFunc) {
        TE_PY_DECREF(PyUploadExceptionFunc);
    }
    PyUploadExceptionFunc = pyUploadFunc;

    return true;
}

void PyWrapper::PrintPyException(bool stderrPrint)
{
    if (PyPrintExceptionFunc == nullptr) {
        HandleManager::Instance().TE_PyErr_Print();
        return;
    }

    PyObject *ptype = nullptr;
    PyObject *pvalue = nullptr;
    PyObject *ptb = nullptr;

    HandleManager::Instance().TE_PyErr_Fetch(&ptype, &pvalue, &ptb);
    HandleManager::Instance().TE_PyErr_NormalizeException(&ptype, &pvalue, &ptb);
    std::string err_info = "";
    PyWrapper::PrintPyException(ptype, pvalue, ptb, err_info, stderrPrint);
}

void PyWrapper::PrintPyException(PyObject *ptype, PyObject *pvalue, PyObject *ptb,
                                 std::string &errInfo, bool stderrPrint)
{
    PyObject *pyTrue = HandleManager::Instance().get_py_true();
    PyObject *pyFalse = HandleManager::Instance().get_py_false();
    PyObject *pyNone = HandleManager::Instance().get_py_none();
    if (pyTrue == nullptr || pyFalse == nullptr || pyNone == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get pyTrue or pyFalse or pyNone from class HandleManager.");
        return;
    }
    if (PyPrintExceptionFunc == nullptr) {
        HandleManager::Instance().TE_PyErr_Print();
    } else {
        PyObject *err = HandleManager::Instance()._PyObject_CallFunction(PyPrintExceptionFunc, "OOOO",
            ptype ? ptype : pyNone,
            pvalue ? pvalue : pyNone,
            ptb ? ptb : pyNone,
            stderrPrint ? pyTrue : pyFalse);
        if (err == nullptr) {
            HandleManager::Instance().TE_PyErr_Print();
            return;
        }
        char *errStr = nullptr;
        int rc = HandleManager::Instance()._PyArg_Parse(err, "s", &errStr);
        if (rc == 0) {
            HandleManager::Instance().TE_PyErr_Print();
            return;
        }
        errInfo = errStr;
        auto iter = errInfo.find("errCode");
        if (iter != errInfo.npos) {
            string errcode = errInfo.substr(iter + 8, 6);
            std::map<std::string, std::string> mapArgs = {{"result", "Common Compilation Python Error"},
                                                          {"reason", "Compiler Error check failed"}};
            TeErrMessageReport(errcode, mapArgs);
        } else {
            TE_FUSION_LOG_FULL(TE_FUSION_LOG_WARNING, "%s", errStr); //lint !e813 !e666
        }
        TE_PY_DECREF(err);
    }
}

void PyWrapper::UploadPyException(const PyObjectPtr &pyModule,
                                  std::map<std::string, std::string> &mapArgs)
{
    if (PyUploadExceptionFunc == nullptr) {
        HandleManager::Instance().TE_PyErr_Print();
        return;
    } else {
        PyObject *pyNone = HandleManager::Instance().get_py_none();
        if (pyNone == nullptr) {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_ERROR, "Failed to get pyNone from class HandleManager.");
            return;
        }
        PyObject *err = HandleManager::Instance()._PyObject_CallFunction(PyUploadExceptionFunc, "OOO",
            pyModule.ptype ? pyModule.ptype : pyNone,
            pyModule.pvalue ? pyModule.pvalue : pyNone,
            pyModule.ptb ? pyModule.ptb : pyNone);
        if (err == nullptr) {
            HandleManager::Instance().TE_PyErr_Print();
            return;
        }
        PyWrapper::PydictToMap(err, mapArgs);
    }
}

void PyWrapper::PyErrMessageReport(const PyObjectPtr &pyRes)
{
    std::map<std::string, std::string> mapArgs = {{"op_name", ""}};
    std::string errorCode;
    UploadPyException(pyRes, mapArgs);
    const std::map<std::string, std::string>::const_iterator &it = mapArgs.find("errCode");
    if (it != mapArgs.end()) {
        errorCode = it->second;
    } else {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "failed to upload errcode message.");
    }
    if (!errorCode.empty()) {
        TeErrMessageReport(errorCode, mapArgs);
    } else {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "Cannot get err code");
    }
}

void PyWrapper::UploadPyException(PyObject *pyObj,
                                  std::vector<std::map<std::string, std::string>> &mapListArgs)
{
    if (pyObj == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "pyErrMsgTuple is nullptr.");
        return;
    }
    Py_ssize_t len = HandleManager::Instance().TE_PyTuple_Size(pyObj);
    if (len != PY_EXCEPT_TUPLE_LEN) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Failed to parse pyErrMsgTuple.");
        return;
    }
    PyObject *pyErrList = HandleManager::Instance().TE_PyTuple_GetItem(pyObj, 0);
    PyObject *pyOpName = HandleManager::Instance().TE_PyTuple_GetItem(pyObj, 1);
    if (pyErrList == nullptr || pyOpName == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "pyErrDict or pyOpName is nullptr.");
        return;
    }
    if (PyList_Check(pyErrList) == 0 || PyUnicode_Check(pyOpName) == 0) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "pyErrMsgTuple info is not true.");
        return;
    }
    const char *temp = HandleManager::Instance().TE_PyUnicode_AsUTF8(pyOpName);
    if (temp == nullptr) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "TE_PyUnicode_AsUTF8(pyOpName) is nullptr!");
    } else {
        std::string OpName = temp;
        std::map<std::string, std::string> mapArgs;
        mapArgs["op_name"] = OpName;
        Py_ssize_t listLen = HandleManager::Instance().TE_PyList_Size(pyErrList);
        for (Py_ssize_t i = 0; i < listLen; ++i) {
            PyObject *pyErrDict = HandleManager::Instance().TE_PyList_GetItem(pyErrList, i);
            PyWrapper::PydictToMap(pyErrDict, mapArgs);
            mapListArgs.emplace_back(mapArgs);
        }
    }
}

void PyWrapper::PydictToMap(PyObject *pyObj,
                            std::map<std::string, std::string> &mapArgs)
{
    if (pyObj == nullptr || PyDict_Check(pyObj) == 0) {
        return;
    }

    PyObject *py_keys = HandleManager::Instance().TE_PyDict_Keys(pyObj);
    if (py_keys == nullptr || PyList_Check(py_keys) == 0) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "py_keys is nullptr or is not List.");
        return;
    }
    Py_ssize_t len = HandleManager::Instance().TE_PyList_Size(py_keys);
    for (int i = 0; i < len; ++i) {
        PyObject* py_key = HandleManager::Instance().TE_PyList_GetItem(py_keys, i);
        PyObject* py_val = HandleManager::Instance().TE_PyDict_GetItem(pyObj, py_key);
        if (py_key == nullptr || PyUnicode_Check(py_key) == 0) {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "py_key is nullptr or is not string!");
            continue;
        }
        if (py_val == nullptr || PyUnicode_Check(py_val) == 0) {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "py_val is nullptr or is not string!");
            continue;
        }
        const char* pKey = HandleManager::Instance().TE_PyUnicode_AsUTF8(py_key);
        const char* pVal = HandleManager::Instance().TE_PyUnicode_AsUTF8(py_val);

        if (pKey != nullptr && pVal != nullptr) {
            std::string map_key = pKey;
            std::string map_val = pVal;
            mapArgs[map_key] = map_val;
        } else {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "pKey or pVal is nullptr!");
        }
    }
}

void PyWrapper::PyObjectDecRef(PyObject *pyObj)
{
    // call Py_XDECREF here result in core dump when python exit,
    // comment out temporary
    if (HandleManager::Instance().TE_Py_IsInitialized() != 0) {
        PyObject *py_decref_tmp = reinterpret_cast<PyObject *>(pyObj);
        if (py_decref_tmp != nullptr) {
            if (--(py_decref_tmp)->ob_refcnt == 0) {
                HandleManager::Instance().TE_Py_Dealloc(py_decref_tmp);
            }
        }
    }
}

PyObjectPtr PyWrapper::ImportModule(const char *moduleName)
{
    if (moduleName == nullptr) {
        return PyObjectPtr();
    }

    PyObject *pyModule = HandleManager::Instance().TE_PyImport_ImportModule(moduleName);
    if (pyModule == nullptr) {
        // python exception occured, just clear the exception
        std::string error_info = "";
        PyObject *ptype = nullptr;
        PyObject *pvalue = nullptr;
        PyObject *ptb = nullptr;

        HandleManager::Instance().TE_PyErr_Fetch(&ptype, &pvalue, &ptb);
        HandleManager::Instance().TE_PyErr_NormalizeException(&ptype, &pvalue, &ptb);
        PrintPyException(ptype, pvalue, ptb, error_info);
        std::map<std::string, std::string> importMapArgs = {{"result", moduleName},
                                                            {"reason", error_info}};
        TeErrMessageReport(EM_IMPORT_PYTHON_FAILED_ERROR, importMapArgs);
        return PyObjectPtr();
    }
    return PyObjectPtr(pyModule, PyWrapper::PyObjectDecRef); //lint !e507
}

int PyWrapper::PyParseStrList(const PyObjectPtr &pyStrList, std::vector<std::string> &vstr)
{
    if (pyStrList == nullptr || PyList_Check(pyStrList.get()) == 0) {
        return -1;
    }

    int nret = 0;
    int listLen = HandleManager::Instance().TE_PyList_Size(pyStrList.get());
    for (int i = 0; i < listLen; ++i) {
        char *str = nullptr;
        PyObject *po = HandleManager::Instance().TE_PyList_GetItem(pyStrList.get(), i);
        int ret = HandleManager::Instance()._PyArg_Parse(po, "s", &str);
        if (ret == 0) {
            continue;
        }

        vstr.emplace_back(str);
        nret++;
    }
    return nret;
}
} // namespace fusion
} // namespace te


