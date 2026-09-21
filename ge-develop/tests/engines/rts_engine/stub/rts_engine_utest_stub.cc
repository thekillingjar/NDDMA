#include "rts_engine_utest_stub.h"
#include <stdarg.h>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <list>
#include <deque>
#include "common/types.h"

using namespace std;

static char g_logHead[] = "ERR!";

namespace ge {
REGISTER_OPTYPE_DEFINE(IDENTITY, "Identity");
REGISTER_OPTYPE_DEFINE(MEMCPYASYNC, "MemcpyAsync");
REGISTER_OPTYPE_DEFINE(MEMCPYADDRASYNC, "MemcpyAddrAsync");
}  // namespace ge

#define MAX_LOG_BUF_SIZE 2048

static char *logLevel[] = {
    (char *)"EMERG",  (char *)"ALERT", (char *)"CRIT", (char *)"ERR",   (char *)"WARNING",
    (char *)"NOTICE", (char *)"EVENT", (char *)"INFO", (char *)"DEBUG", (char *)"RESERVED",
};

int CheckLogLevel(int moduleId, int logLevel) {
  return 0;
}

void DlogRecord(int moduleId, int level, const char *fmt, ...) {
  char buf[MAX_LOG_BUF_SIZE] = {0};

  va_list arg;
  va_start(arg, fmt);
  vsnprintf(buf, MAX_LOG_BUF_SIZE, fmt, arg);
  va_end(arg);

  return;
}

void DlogErrorInner(int module_id, const char *fmt, ...) {
  char buf[MAX_LOG_BUF_SIZE] = {0};

  va_list arg;
  va_start(arg, fmt);
  vsnprintf(buf, MAX_LOG_BUF_SIZE, fmt, arg);
  va_end(arg);

  return;
}

void DlogWarnInner(int module_id, const char *fmt, ...) {
  char buf[MAX_LOG_BUF_SIZE] = {0};

  va_list arg;
  va_start(arg, fmt);
  vsnprintf(buf, MAX_LOG_BUF_SIZE, fmt, arg);
  va_end(arg);

  return;
}

void DlogInfoInner(int module_id, const char *fmt, ...) {
  char buf[MAX_LOG_BUF_SIZE] = {0};

  va_list arg;
  va_start(arg, fmt);
  vsnprintf(buf, MAX_LOG_BUF_SIZE, fmt, arg);
  va_end(arg);

  return;
}

void DlogDebugInner(int module_id, const char *fmt, ...) {
  char buf[MAX_LOG_BUF_SIZE] = {0};

  va_list arg;
  va_start(arg, fmt);
  vsnprintf(buf, MAX_LOG_BUF_SIZE, fmt, arg);
  va_end(arg);

  return;
}

void DlogEventInner(int module_id, const char *fmt, ...) {
  char buf[MAX_LOG_BUF_SIZE] = {0};

  va_list arg;
  va_start(arg, fmt);
  vsnprintf(buf, MAX_LOG_BUF_SIZE, fmt, arg);
  va_end(arg);

  return;
}

int dlog_getlevel(int module_id, int *enable_event) {
  if (enable_event != nullptr) {
    *enable_event = true;
    return 0;
  }
}

int rtSetTaskGenCallback_stub_with_notify(rtTaskGenCallback callback) {
  void *model;
  rtTaskInfo_t task;
  task.type = RT_MODEL_TASK_MEMCPY_ASYNC;
  callback(model, &task);

  task.type = RT_MODEL_TASK_LABEL_SET;
  callback(model, &task);

  task.type = RT_MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX;
  callback(model, &task);

  task.type = RT_MODEL_TASK_STREAM_SWITCH;
  callback(model, &task);

  task.type = RT_MODEL_TASK_NPU_GET_FLOAT_STATUS;
  callback(model, &task);

  task.type = RT_MODEL_TASK_NPU_CLEAR_FLOAT_STATUS;
  callback(model, &task);

  task.type = RT_MODEL_TASK_NPU_GET_DEBUG_FLOAT_STATUS;
  callback(model, &task);

  task.type = RT_MODEL_TASK_NPU_CLEAR_DEBUG_FLOAT_STATUS;
  callback(model, &task);

  return 0;
}

int rtSetTaskGenCallback_stub_with_memcpyaddr(rtTaskGenCallback callback) {
  void *model;
  rtTaskInfo_t task;

  task.type = RT_MODEL_TASK_MEMCPY_ADDR_ASYNC;
  callback(model, &task);

  return 0;
}

int rtSetTaskGenCallback_stub(rtTaskGenCallback callback) {
  void *model;
  rtTaskInfo_t task;

  task.type = RT_MODEL_TASK_NOTIFY_RECORD;
  callback(model, &task);

  task.type = RT_MODEL_TASK_NOTIFY_WAIT;
  callback(model, &task);

  return 0;
}

int rtSetTaskGenCallback_stub_with_cmoAddr(rtTaskGenCallback callback) {
  void *model;
  rtTaskInfo_t task;

  task.type = RT_MODEL_TASK_CMO_ADDR;
  callback(model, &task);
  return 0;
}

namespace {
std::string STAGES_STR = "[TEST][TEST]";
}

ErrorManager &ErrorManager::GetInstance() {
  static ErrorManager instance;
  return instance;
}

void ErrorManager::SetStage(const std::string &firstStage, const std::string &secondStage) {
  STAGES_STR = '[' + firstStage + "][" + secondStage + ']';
}

const string &ErrorManager::GetLogHeader() {
  return STAGES_STR;
}

int ErrorManager::Init() {
  return 0;
}

void ErrorManager::ATCReportErrMessage(std::string error_code, const std::vector<std::string> &key,
                                       const std::vector<std::string> &value) {
  return;
}

std::string ErrorManager::GetErrorMessage() {
  std::string message = "";
  return message;
}

int ErrorManager::ReportInterErrMessage(std::string error_code, const std::string &error_msg) {
  return 0;
}

int error_message::FormatErrorMessage(char *str_dst, size_t dst_max, const char *format, ...) {
  return 1;
}

void error_message::ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line,
                                     const std::string error_code, const char_t *format, ...) {
  return;
}

#ifdef __GNUC__
std::string error_message::TrimPath(const std::string &str) {
  if (str.find_last_of('/') != std::string::npos) {
    return str.substr(str.find_last_of('/') + 1U);
  }
  return str;
}
#else
std::string error_message::TrimPath(const std::string &str) {
  if (str.find_last_of('\\') != std::string::npos) {
    return str.substr(str.find_last_of('\\') + 1U);
  }
  return str;
}
#endif
