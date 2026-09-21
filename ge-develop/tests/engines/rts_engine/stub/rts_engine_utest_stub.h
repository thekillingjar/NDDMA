#ifndef RTS_ENGINE_UTEST_STUB
#define RTS_ENGINE_UTEST_STUB

#include <string>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <list>
#include <deque>
#include "runtime/dev.h"
#include "runtime/rt_model.h"

using namespace std;

namespace error_message {
std::string TrimPath(const std::string &str);
#ifdef __GNUC__
int FormatErrorMessage(char *str_dst, size_t dst_max, const char *format, ...) __attribute__((format(printf, 3, 4)));
#define TRIM_PATH(x) strrchr(x, '/') ? strrchr(x, '/') + 1 : x
void ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line, const std::string error_code,
                      const char_t *format, ...) __attribute__((format(printf, 5, 6)));
#else
int FormatErrorMessage(char *str_dst, size_t dst_max, const char *format, ...);
#define TRIM_PATH(x) strrchr(x, '\\') ? strrchr(x, '\\') + 1 : x
void ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line, const std::string error_code,
                      const char_t *format, ...);
#endif
}  // namespace error_message

class ErrorManager {
 public:
  static ErrorManager &GetInstance();
  void SetStage(const std::string &firstStage, const std::string &secondStage);
  void ATCReportErrMessage(std::string error_code, const std::vector<std::string> &key = {},
                           const std::vector<std::string> &value = {});
  int Init();
  int ReportInterErrMessage(std::string error_code, const std::string &error_msg);
  std::string GetErrorMessage();
  const std::string &GetLogHeader();
};

int rtSetTaskGenCallback_stub(rtTaskGenCallback callback);
int rtSetTaskGenCallback_stub_with_notify(rtTaskGenCallback callback);
int rtSetTaskGenCallback_stub_with_cmoAddr(rtTaskGenCallback callback);
int rtSetTaskGenCallback_stub_with_memcpyaddr(rtTaskGenCallback callback);

#endif  // _RTS_ENGINE_UTEST_STUB_