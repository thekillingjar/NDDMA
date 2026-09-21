/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "process_utils.h"

#include "base/err_msg.h"

#include <unistd.h>
#include <fstream>
#include <cstdlib>
#include <csignal>
#include <regex>
#include <arpa/inet.h>
#include <net/if.h>
#include <sys/socket.h>
#include "common/checker.h"
#include "common/debug/ge_log.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr char_t kValidChars[] = "[A-Za-z\\d/_.-]+";
constexpr char_t kInvalidChars[] = "[^A-Za-z\\d/_.-]+";
}
pid_t ProcessUtils::Fork() {
  return fork();
}

int32_t ProcessUtils::Execute(const std::string &path, char_t *const *argv) {
  return execv(path.c_str(), argv);
}

Status ProcessUtils::System(const std::string &cmd, bool print_err) {
  sighandler_t old_handler = signal(SIGCHLD, SIG_DFL);
  GE_MAKE_GUARD(old_handler, [&old_handler]() { signal(SIGCHLD, old_handler); });
  auto status = system(cmd.c_str());
  if (status == -1) {
    if (print_err) {
      GELOGE(FAILED, "Failed to execute cmd.");
    }
    return FAILED;
  }

  if (WIFEXITED(status) && (WEXITSTATUS(status) == 0)) {
    return SUCCESS;
  }

  if (print_err) {
    GELOGE(FAILED, "Execute cmd result failed, ret = %d.", WEXITSTATUS(status));
  }
  return FAILED;
}

Status ProcessUtils::GetIpaddr(const std::string &name, std::string &ipaddr) {
  auto socket_fd = mmSocket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd == EN_ERROR) {
    GELOGE(FAILED, "[Get][Fd] Get socket fd error.");
    return FAILED;
  }

  const size_t kBufsize = 1024U;
  char_t buf[kBufsize]{};
  struct ifconf ifc;
  ifc.ifc_len = kBufsize;
  ifc.ifc_buf = buf;
  mmIoctlBuf ioctl_buf;
  ioctl_buf.inbuf = &ifc;
  auto ret = mmIoctl(socket_fd, SIOCGIFCONF, &ioctl_buf);
  if (ret != EN_OK) {
    GELOGE(FAILED, "[Get][Ifconfig] Get ifconfig info error.");
    return FAILED;
  }

  struct ifreq *ifr = reinterpret_cast<struct ifreq *>(buf);
  for (size_t i = 0U; i < ifc.ifc_len / sizeof(struct ifreq); ++i) {
    struct sockaddr_in *addr = reinterpret_cast<struct sockaddr_in *>(&(ifr->ifr_addr));
    char_t ip[INET_ADDRSTRLEN]{};
    auto addr_ret = inet_ntop(AF_INET, &(addr->sin_addr), ip, INET_ADDRSTRLEN);
    GE_CHECK_NOTNULL(addr_ret);
    std::string r_name = ifr->ifr_name;
    if  (r_name != name) {
      ifr += 1;
      continue;
    }
    ipaddr = ip;
    GELOGI("[Get][Ip] Get ipaddr success, ifr name = %s, ip = %s.", ifr->ifr_name, ip);
    return SUCCESS;
  }
  GELOGE(FAILED, "[Get][Ip] Get ipaddr failed, net name = %s", name.c_str());
  return FAILED;
}

pid_t ProcessUtils::WaitPid(pid_t pid, int32_t *wstatus, int options) {
  return waitpid(pid, wstatus, options);
}

Status ProcessUtils::DoMkdir(const char_t *tmp_dir_path, mmMode_t mode) {
  if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
    const int32_t ret = mmMkdir(tmp_dir_path, mode);
    if (ret != 0) {
      auto create_errno = errno;
      // may create dir multi thread
      if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
        REPORT_INNER_ERR_MSG("E18888",
                          "Can not create directory %s. Make sure the directory "
                          "exists and writable. errmsg:%s",
                          tmp_dir_path, GetErrorNumStr(create_errno).c_str());
        GELOGE(FAILED, "Create directory %s failed, reason:%s. Make sure the "
                      "directory exists and writable.",
              tmp_dir_path, GetErrorNumStr(create_errno).c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status ProcessUtils::CreateDir(const std::string &directory_path) {
  constexpr auto mode = static_cast<uint32_t>(M_IRUSR | M_IWUSR | M_IXUSR);
  return ProcessUtils::CreateDir(directory_path, mode);
}

Status ProcessUtils::CreateDir(const std::string &directory_path, uint32_t mode) {
  if (directory_path.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "directory path is empty, check invalid");
    GELOGE(FAILED, "[Check][Param] directory path is empty.");
    return FAILED;
  }

  const auto dir_path_len = directory_path.length();
  if (dir_path_len >= static_cast<size_t>(MMPA_MAX_PATH)) {
    REPORT_PREDEFINED_ERR_MSG("E13002", std::vector<const char *>({"filepath", "size"}),
                              std::vector<const char *>({directory_path.c_str(), std::to_string(MMPA_MAX_PATH).c_str()}));
    GELOGE(FAILED, "Path %s len is too long, it must be less than %d", directory_path.c_str(), MMPA_MAX_PATH);
    return FAILED;
  }
  char_t tmp_dir_path[MMPA_MAX_PATH] = {};
  const auto mkdir_mode = static_cast<mmMode_t>(mode);
  for (size_t i = 0U; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      GE_CHK_STATUS_RET(DoMkdir(&(tmp_dir_path[0U]), mkdir_mode), "Failed to mkdir");
    }
  }
  return DoMkdir(directory_path.c_str(), mkdir_mode);
}

Status ProcessUtils::IsValidPath(const std::string &path) {
  GE_CHK_BOOL_RET_STATUS((path.find("..") == std::string::npos),
                         FAILED,
                         "File path[%s] is invalid, include relative path.",
                         path.c_str());
  std::regex e(kValidChars);
  std::smatch sm;
  GE_CHK_BOOL_RET_STATUS(std::regex_match(path, sm, e), FAILED,
                         "Path[%s] is invalid, please use a-z, A-Z, 0-9, _, - and /",
                         path.c_str());
  return SUCCESS;
}

std::string ProcessUtils::NormalizePath(const std::string &path) {
  std::regex e(kInvalidChars);
  return std::regex_replace(path, e, "_");
}
}  // namespace ge
