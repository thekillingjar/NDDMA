/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mmpa_stub.h"
#include <dirent.h>
#include "mmpa/mmpa_api.h"
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

class ComputeGraph;
#define MMPA_MAX_SLEEP_MILLSECOND_USING_USLEEP 1000
#define MMPA_MSEC_TO_USEC 1000
#define MMPA_MAX_SLEEP_MICROSECOND_USING_USLEEP 1000000

INT32 mmOpen(const CHAR *path_name, INT32 flags) {
  INT32 fd = HANDLE_INVALID_VALUE;

  if (nullptr == path_name) {
    syslog(LOG_ERR, "The path name pointer is null.\r\n");
    return EN_INVALID_PARAM;
  }
  if ((flags != O_RDONLY) && (0 == (flags & (O_WRONLY | O_RDWR | O_CREAT)))) {
    syslog(LOG_ERR, "The file open mode is error.\r\n");
    return EN_INVALID_PARAM;
  }

  fd = ge::MmpaStub::GetInstance().GetImpl()->Open2(path_name, flags, S_IRWXU | S_IRWXG);
  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "Open file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return fd;
}

INT32 mmOpen2(const CHAR *path_name, INT32 flags, MODE mode) {
  INT32 fd = HANDLE_INVALID_VALUE;

  if (nullptr == path_name) {
    syslog(LOG_ERR, "The path name pointer is null.\r\n");
    return EN_INVALID_PARAM;
  }
  if (MMPA_ZERO == (flags & (O_RDONLY | O_WRONLY | O_RDWR | O_CREAT))) {
    syslog(LOG_ERR, "The file open mode is error.\r\n");
    return EN_INVALID_PARAM;
  }
  if ((MMPA_ZERO == (mode & (S_IRUSR | S_IREAD))) && (MMPA_ZERO == (mode & (S_IWUSR | S_IWRITE)))) {
    syslog(LOG_ERR, "The permission mode of the file is error.\r\n");
    return EN_INVALID_PARAM;
  }

  fd = ge::MmpaStub::GetInstance().GetImpl()->Open2(path_name, flags, mode);
  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "Open file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return fd;
}

INT32 mmClose(INT32 fd) {
  INT32 result = EN_OK;

  if (fd < MMPA_ZERO) {
    syslog(LOG_ERR, "The file fd is invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = close(fd);
  if (EN_OK != result) {
    syslog(LOG_ERR, "Close the file failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return EN_OK;
}

mmSsize_t mmWrite(INT32 fd, VOID *mm_buf, UINT32 mm_count) {
  return ge::MmpaStub::GetInstance().GetImpl()->Write(fd, mm_buf, mm_count);
}

mmSsize_t mmRead(INT32 fd, VOID *mm_buf, UINT32 mm_count) {
  mmSsize_t result = MMPA_ZERO;

  if ((fd < MMPA_ZERO) || (nullptr == mm_buf)) {
    syslog(LOG_ERR, "Input parameter invalid.\r\n");
    return EN_INVALID_PARAM;
  }

  result = read(fd, mm_buf, (size_t)mm_count);
  if (result < MMPA_ZERO) {
    syslog(LOG_ERR, "Read file to buf failed, errno is %s.\r\n", strerror(errno));
    return EN_ERROR;
  }
  return result;
}

VOID *mmMmap(mmFd_t fd, mmSize_t size, mmOfft_t offset, mmFd_t *extra, INT32 prot, INT32 flags) {
  return mmap(nullptr, size, prot, flags, fd, offset);
}

INT32 mmMunMap(VOID *data, mmSize_t size, mmFd_t *extra) {
  return munmap(data, size);
}

INT32 mmFStatGet(INT32 fd, mmStat_t *buf) {
  return ge::MmpaStub::GetInstance().GetImpl()->FStatGet(fd, buf);
}

INT32 mmMkdir(const CHAR *lp_path_name, mmMode_t mode) {
  return ge::MmpaStub::GetInstance().GetImpl()->mmMkdir(lp_path_name, mode);
}

INT32 mmRmdir(const CHAR *lp_path_name) {
  INT32 ret;
  DIR *childDir = nullptr;

  if (lp_path_name == nullptr) {
    return EN_INVALID_PARAM;
  }
  DIR *dir = opendir(lp_path_name);
  if (dir == nullptr) {
    return EN_INVALID_PARAM;
  }

  const struct dirent *entry = nullptr;
  size_t bufSize = strlen(lp_path_name) + (size_t)(PATH_SIZE + 2); // make sure the length is large enough
  while ((entry = readdir(dir)) != nullptr) {
    if ((strcmp(".", entry->d_name) == MMPA_ZERO) || (strcmp("..", entry->d_name) == MMPA_ZERO)) {
      continue;
    }
    CHAR *buf = (CHAR *)malloc(bufSize);
    if (buf == nullptr) {
      break;
    }
    ret = memset_s(buf, bufSize, 0, bufSize);
    if (ret == EN_ERROR) {
      free(buf);
      buf = nullptr;
      break;
    }
    ret = snprintf_s(buf, bufSize, bufSize - 1U, "%s/%s", lp_path_name, entry->d_name);
    if (ret == EN_ERROR) {
      free(buf);
      buf = nullptr;
      break;
    }

    childDir = opendir(buf);
    if (childDir != nullptr) {
      (VOID)closedir(childDir);
      (VOID)mmRmdir(buf);
      free(buf);
      buf = nullptr;
      continue;
    } else {
      ret = unlink(buf);
      if (ret == EN_OK) {
        free(buf);
        continue;
      }
    }
    free(buf);
    buf = nullptr;
  }
  (VOID)closedir(dir);

  ret = rmdir(lp_path_name);
  if (ret == EN_ERROR) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmChmod(const CHAR *filename, INT32 mode) {
  return chmod(filename, mode);
}

mmTimespec mmGetTickCount() {
  mmTimespec rts;
  struct timespec ts = {0};
  (void)clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  rts.tv_sec = ts.tv_sec;
  rts.tv_nsec = ts.tv_nsec;
  return rts;
}

INT32 mmGetTid() {
  INT32 ret = (INT32)syscall(SYS_gettid);

  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }

  return ret;
}

INT32 mmGetSystemTime(mmSystemTime_t *sysTime) {
  return ge::MmpaStub::GetInstance().GetImpl()->mmGetSystemTime(sysTime);
}

INT32 mmAccess(const CHAR *path_name) {
  return ge::MmpaStub::GetInstance().GetImpl()->Access(path_name);
}

INT32 mmStatGet(const CHAR *path, mmStat_t *buffer) {
  return ge::MmpaStub::GetInstance().GetImpl()->StatGet(path, buffer);
}

INT32 mmGetFileSize(const CHAR *file_name, ULONGLONG *length) {
  if ((file_name == nullptr) || (length == nullptr)) {
    return EN_INVALID_PARAM;
  }
  struct stat file_stat;
  (void)memset_s(&file_stat, sizeof(file_stat), 0, sizeof(file_stat));  // unsafe_function_ignore: memset
  INT32 ret = lstat(file_name, &file_stat);
  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }
  *length = (ULONGLONG)file_stat.st_size;
  return EN_OK;
}

INT32 mmScandir(const CHAR *path, mmDirent ***entryList, mmFilter filterFunc,  mmSort sort)
{
  if ((path == nullptr) || (entryList == nullptr)) {
    return EN_INVALID_PARAM;
  }
  INT32 count = scandir(path, entryList, filterFunc, sort);
  if (count < MMPA_ZERO) {
    return EN_ERROR;
  }
  return count;
}

VOID mmScandirFree(mmDirent **entryList, INT32 count)
{
  if (entryList == nullptr) {
    return;
  }
  INT32 j;
  for (j = 0; j < count; j++) {
    if (entryList[j] != nullptr) {
      free(entryList[j]);
      entryList[j] = nullptr;
    }
  }
  free(entryList);
}

INT32 mmAccess2(const CHAR *pathName, INT32 mode)
{
  return ge::MmpaStub::GetInstance().GetImpl()->mmAccess2(pathName, mode);
}

INT32 mmGetTimeOfDay(mmTimeval *timeVal, mmTimezone *timeZone) {
  return ge::MmpaStub::GetInstance().GetImpl()->mmGetTimeOfDay(timeVal, timeZone);
}

INT32 mmRealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen)
{
  if (path == nullptr || realPath == nullptr || realPathLen < MMPA_MAX_PATH) {
    return EN_INVALID_PARAM;
  }

  std::string str_path = path;
  if (str_path.find("libcce.so") != std::string::npos) {
    strncpy(realPath, path, realPathLen);
    return EN_OK;
  }

  return ge::MmpaStub::GetInstance().GetImpl()->RealPath(path, realPath, realPathLen);
}

INT32 mmRWLockInit(mmRWLock_t *rwLock)
{
  if (rwLock == nullptr) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_init(rwLock, nullptr);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockRDLock(mmRWLock_t *rwLock)
{
  if (rwLock == nullptr) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_rdlock(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockWRLock(mmRWLock_t *rwLock)
{
  if (rwLock == nullptr) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_wrlock(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRDLockUnLock(mmRWLock_t *rwLock)
{
  if (rwLock == nullptr) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_unlock(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmWRLockUnLock(mmRWLock_t *rwLock)
{
  if (rwLock == nullptr) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_unlock(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockDestroy(mmRWLock_t *rwLock)
{
  if (rwLock == nullptr) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_destroy(rwLock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmGetErrorCode()
{
  return errno;
}

INT32 mmIsDir(const CHAR *fileName)
{
  if (fileName == nullptr) {
    return EN_ERR;
  }

  DIR *pDir = opendir (fileName);
  if (pDir != nullptr) {
    (void) closedir (pDir);
    return EN_OK;
  }
  return EN_ERR;
}

INT32 mmGetEnv(const CHAR *name, CHAR *value, UINT32 len)
{
  const char *env = getenv(name);
  if (env == nullptr) {
    return EN_ERROR;
  }

  strncpy(value, env, len);
  return EN_OK;
}

CHAR *mmDlerror() {
  return dlerror();
}

INT32 mmDladdr(VOID *addr, mmDlInfo *info) {
  int ret = dladdr(addr, (Dl_info *)info);
  if (ret != -1){
    return 0;
  }
  return -1;
}

VOID *mmDlopen(const CHAR *fileName, INT32 mode) {
  return ge::MmpaStub::GetInstance().GetImpl()->DlOpen(fileName, mode);
}

INT32 mmDlclose(VOID *handle) {
  if (handle == nullptr){
    return 1;
  }
  if (handle == (void *)0x8888) {
    return 0;
  }

  return ge::MmpaStub::GetInstance().GetImpl()->DlClose(handle);
}

VOID *mmDlsym(VOID *handle, const CHAR *funcName) {
  if (reinterpret_cast<uintptr_t>(handle) < 0x8000) {
    return nullptr;
  }

  return ge::MmpaStub::GetInstance().GetImpl()->DlSym(handle, funcName);
}

INT32 mmGetPid()
{
  return (INT32)getpid();
}

INT32 mmSetCurrentThreadName(const CHAR *name)
{
  return EN_OK;
}

INT32 mmGetCwd(CHAR *buffer, INT32 maxLen)
{
  return EN_OK;
}

CHAR *mmGetErrorFormatMessage(int errnum, CHAR *buf, mmSize size)
{
  if ((buf == nullptr) || (size <= 0)) {
    return nullptr;
  }
  return strerror_r(errnum, buf, size);
}

CHAR *mmDirName(CHAR *path) {
  if (path == nullptr) {
    return nullptr;
  }
#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
  char separator = '\\';
#else
  char separator = '/';
#endif
  std::string path_str(path);
  const size_t last_sep_pos = path_str.rfind(separator);
  if (last_sep_pos == std::string::npos) {
    return nullptr;
  }

  path[last_sep_pos] = '\0';
  return path;
}

INT32 mmCreateTask(mmThread *threadHandle, mmUserBlock_t *funcBlock) {
  if ((threadHandle == nullptr) || (funcBlock == nullptr) || (funcBlock->procFunc == nullptr)) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_create(threadHandle, nullptr, funcBlock->procFunc, funcBlock->pulArg);
  if (ret != EN_OK) {
    ret = EN_ERROR;
  }
  return ret;
}

INT32 mmJoinTask(mmThread *threadHandle) {
  if (threadHandle == nullptr) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_join(*threadHandle, nullptr);
  if (ret != EN_OK) {
    ret = EN_ERROR;
  }
  return ret;
}

INT32 mmSleep(UINT32 millSecond) {
  if (millSecond == MMPA_ZERO) {
    return EN_INVALID_PARAM;
  }
  UINT32 microSecond;

  if (millSecond <= MMPA_MAX_SLEEP_MILLSECOND_USING_USLEEP) {
    microSecond = millSecond * (UINT32)MMPA_MSEC_TO_USEC;
  } else {
    microSecond = MMPA_MAX_SLEEP_MICROSECOND_USING_USLEEP;
  }
  return ge::MmpaStub::GetInstance().GetImpl()->Sleep(microSecond);
}

INT32 mmUnlink(const CHAR *filename) {
  if (filename == nullptr) {
    return EN_INVALID_PARAM;
  }
  return unlink(filename);
}

INT32 mmSetEnv(const CHAR *name, const CHAR *value, INT32 overwrite) {
  if ((name == nullptr) || (value == nullptr)) {
    return EN_INVALID_PARAM;
  }
  return setenv(name, value, overwrite);
}

INT32 mmWaitPid(mmProcess pid, INT32 *status, INT32 options) {
  return ge::MmpaStub::GetInstance().GetImpl()->WaitPid(pid, status, options);
}

VOID mmSetOptErr(INT32 mmOptErr) {
  opterr = mmOptErr;
}

INT32 mmGetOptLong(INT32 argc, CHAR * const * argv, const CHAR *opts, const mmStructOption *longOpts, INT32 *longIndex) {
  return getopt_long(argc, argv, opts, longOpts, longIndex);
}

CHAR* mmGetOptArg(VOID) {
  return optarg;
}

INT32 mmGetOptInd(VOID) {
  return optind;
}

INT32 mmUmask(INT32 pmode) {
  return 0;
}

mmSockHandle mmSocket(INT32 sockFamily, INT32 type, INT32 protocol) {
  return EN_ERROR;
}

INT32 mmIoctl(mmProcess fd, INT32 ioctlCode, mmIoctlBuf *bufPtr) {
  return 0;
}

ge::MmpaStub& ge::MmpaStub::GetInstance() {
  static ge::MmpaStub instance;
  return instance;
}

void ge::MmpaStub::SetImpl(const std::shared_ptr<ge::MmpaStubApiGe> &impl) {
  impl_ = impl;
}

ge::MmpaStubApiGe* ge::MmpaStub::GetImpl() {
  return impl_.get();
}

void ge::MmpaStub::Reset() {
  impl_ = std::make_shared<ge::MmpaStubApiGe>();
}
#ifdef __cplusplus
}
#endif
