/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYWRAPPER_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYWRAPPER_H_

#include <Python.h>
#include <map>
#include <memory>
#include <vector>

#include "inc/te_fusion_log.h"
#include "common/common_utils.h"
#include "python_adapter/py_decouple.h"

#define AUTO_PY_DECREF(obj)                     \
    AUTO_PY_DECREF_UNIQ_HELPER(obj, __COUNTER__)

#define AUTO_PY_DECREF_UNIQ_HELPER(obj, counter)    \
    AUTO_PY_DECREF_UNIQ(obj, counter)

#define AUTO_PY_DECREF_UNIQ(obj, counter)                        \
    PyObjectPtr tmpPyObj__##counter(obj, PyWrapper::PyObjectDecRef)

#define TE_PY_DECREF(obj)                                               \
    do {                                                                \
        PyObject *py_decref_tmp = reinterpret_cast<PyObject *>(obj);    \
        if (py_decref_tmp != nullptr) {                                 \
            if ((--(py_decref_tmp)->ob_refcnt) == 0) {                  \
                HandleManager::Instance().TE_Py_Dealloc(py_decref_tmp); \
            }                                                           \
        }                                                               \
    } while(0)


namespace te {
namespace fusion {
using SharedPyObjectPtr = std::shared_ptr<PyObject>;
class PyObjectPtr : public SharedPyObjectPtr {
public:
    PyObjectPtr() = default;
    PyObjectPtr(PyObject* ptr, void(*d)(PyObject*)): std::shared_ptr<PyObject>(ptr, d) {};
    ~PyObjectPtr() = default;
    PyObject *ptype = nullptr;
    PyObject *pvalue = nullptr;
    PyObject *ptb = nullptr;
};

struct PyWrapper {
    static PyObject *PyPrintExceptionFunc;
    static void PrintPyException(bool stderrPrint = true);
    static PyObject *PyUploadExceptionFunc;
    static void PrintPyException(PyObject *ptype, PyObject *pvalue, PyObject *ptb,
                                 std::string &errInfo, bool stderrPrint = true);
    static void PyErrMessageReport(const PyObjectPtr &pyRes);
    static void UploadPyException(const PyObjectPtr &pyModule,
                                  std::map<std::string, std::string> &mapArgs);
    static void UploadPyException(PyObject *pyObj,
                                  std::vector<std::map<std::string, std::string>> &mapListArgs);
    static void PydictToMap(PyObject *pyObj,
                            std::map<std::string, std::string> &mapArgs);
    static bool InitPyLogger();
    static void PyObjectDecRef(PyObject *pyObj);
    static PyObjectPtr ImportModule(const char *moduleName);
    static int PyParseStrList(const PyObjectPtr &pyStrList, std::vector<std::string> &vstr);

    template<typename... Args>
    static PyObjectPtr CallMethodImpl(bool stderrPrint, const PyObjectPtr &pyObj, const char *methodName,
                                      const char *format, Args... args)
    {
        if (pyObj == nullptr) {
            return PyObjectPtr();
        }

        PyObject *pyRes = nullptr;

        if (methodName == nullptr) {
            pyRes = HandleManager::Instance()._PyObject_CallFunction(pyObj.get(), format, args...);
        } else {
            pyRes = HandleManager::Instance().TE_PyObject_CallMethod_SizeT(pyObj.get(), methodName, format, args...);
        }

        if (pyRes == nullptr) {
            // python exception occured, just clear the exception
            PyObject *ptype = nullptr;
            PyObject *pvalue = nullptr;
            PyObject *ptb = nullptr;

            HandleManager::Instance().TE_PyErr_Fetch(&ptype, &pvalue, &ptb);
            HandleManager::Instance().TE_PyErr_NormalizeException(&ptype, &pvalue, &ptb);
            PyObjectPtr res(pyRes, PyWrapper::PyObjectDecRef);
            res.ptype = ptype;
            res.pvalue = pvalue;
            res.ptb = ptb;
            std::string err_info = "";
            PrintPyException(ptype, pvalue, ptb, err_info, stderrPrint);
            if (stderrPrint) {
                std::string result = methodName != nullptr ? methodName : format;
                std::map<std::string, std::string> importMapArgs = {{"func_name", result},
                                                                    {"reason", err_info.c_str()}};
                TeErrMessageReport(EM_CALL_FUNC_MATHOD_ERROR, importMapArgs);
            }
            return res;
        }

        return PyObjectPtr(pyRes, PyWrapper::PyObjectDecRef);
    }

    template<typename... Args>
    static PyObjectPtr CallMethod(const PyObjectPtr &pyObj, const char *methodName,
                                  const char *format, Args... args)
    {
        return CallMethodImpl(true, pyObj, methodName, format, args...);
    }

    template<typename... Args>
    static PyObjectPtr CallMethodNoErrPrint(const PyObjectPtr &pyObj, const char *methodName,
                                            const char *format, Args... args)
    {
        return CallMethodImpl(false, pyObj, methodName, format, args...);
    }

    template<typename... Args>
    static PyObjectPtr CallFunction(const PyObjectPtr &pyFunc, const char *format, Args... args)
    {
        return CallMethod(pyFunc, nullptr, format, args...);
    }

    static PyObjectPtr Call(const PyObjectPtr &pyObj, const char *methodName, PyObject *args, PyObject *kwargs)
    {
        if (pyObj == nullptr) {
            return PyObjectPtr();
        }

        PyObject *tmpArgs = args;
        if (tmpArgs == nullptr) {
            tmpArgs = HandleManager::Instance().TE_PyList_New(0);
            if (tmpArgs == nullptr) {
                return PyObjectPtr();
            }
        }

        PyObject* pyFunc = HandleManager::Instance().TE_PyObject_GetAttrString(pyObj.get(), methodName);
        if (pyFunc == nullptr) {
            return PyObjectPtr();
        }

        PyObject *pyRes = nullptr;
        pyRes = HandleManager::Instance().TE_PyObject_Call(pyFunc, tmpArgs, kwargs);
        if (pyRes == nullptr) {
            // python exception occured, just clear the exception
            PrintPyException();
        }
        return PyObjectPtr(pyRes, PyWrapper::PyObjectDecRef);
    }
};

class PyLockGIL {
public:
    PyLockGIL() : gil(HandleManager::Instance().TE_PyGILState_Ensure())
    {
        TE_DBGLOG("Get py GIL lock");
    }

    ~PyLockGIL()
    {
        HandleManager::Instance().TE_PyGILState_Release(gil);
        TE_DBGLOG("Release py GIL lock");
    }

private:
    PyGILState_STATE gil;
};
} // namespace fusion
} // namespace te
#endif // ATC_OPCOMPILER_TE_FUSION_SOURCE_PYTHON_ADAPTER_PYWRAPPER_H_
