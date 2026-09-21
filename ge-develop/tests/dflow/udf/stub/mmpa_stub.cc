/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mmpa/mmpa_api.h"
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

INT32 mmRealPath(const CHAR *path, CHAR *real_path, INT32 real_path_len) {
  if (path == nullptr || real_path == nullptr || real_path_len < MMPA_MAX_PATH) {
    return EN_INVALID_PARAM;
  }

  strncpy_s(real_path, real_path_len, path, real_path_len);
  return EN_OK;
}

INT32 mmOpen(const CHAR *path_name, INT32 flags) {
  return open(path_name, flags);
}

INT32 mmOpen2(const CHAR *path_name, INT32 flags, MODE mode) {
  return open(path_name, flags, mode);
}

INT32 mmClose(INT32 fd) {
  return close(fd);
}

mmSsize_t mmWrite(INT32 fd, VOID *mm_buf, UINT32 mm_count) {
  return write(fd, mm_buf, mm_count);
}

mmSsize_t mmRead(INT32 fd, VOID *mm_buf, UINT32 mm_count) {
  return read(fd, mm_buf, mm_count);
}

INT32 mmMkdir(const CHAR *lp_path_name, mmMode_t mode) {
  return mkdir(lp_path_name, mode);
}

INT32 mmRmdir(const CHAR *path_name) {
  INT32 ret;
  DIR *child_dir = NULL;

  if (path_name == NULL) {
    return EN_INVALID_PARAM;
  }
  DIR *dir = opendir(path_name);
  if (dir == NULL) {
    return EN_INVALID_PARAM;
  }

  const struct dirent *entry = NULL;
  size_t buf_size = strlen(path_name) + (size_t)(PATH_SIZE + 2);  // make sure the length is large enough
  while ((entry = readdir(dir)) != NULL) {
    if ((strcmp(".", entry->d_name) == MMPA_ZERO) || (strcmp("..", entry->d_name) == MMPA_ZERO)) {
      continue;
    }
    CHAR *buf = (CHAR *)malloc(buf_size);
    if (buf == NULL) {
      break;
    }
    ret = memset_s(buf, buf_size, 0, buf_size);
    if (ret == EN_ERROR) {
      free(buf);
      buf = NULL;
      break;
    }
    ret = snprintf_s(buf, buf_size, buf_size - 1U, "%s/%s", path_name, entry->d_name);
    if (ret == EN_ERROR) {
      free(buf);
      buf = NULL;
      break;
    }

    child_dir = opendir(buf);
    if (child_dir != NULL) {
      (VOID) closedir(child_dir);
      (VOID) mmRmdir(buf);
      free(buf);
      buf = NULL;
      continue;
    } else {
      ret = unlink(buf);
      if (ret == EN_OK) {
        free(buf);
        continue;
      }
    }
    free(buf);
    buf = NULL;
  }
  (VOID) closedir(dir);

  ret = rmdir(path_name);
  if (ret == EN_ERROR) {
    return EN_ERROR;
  }
  return EN_OK;
}

mmTimespec mmGetTickCount() {
  return {};
}

INT32 mmGetTid() {
  INT32 ret = (INT32)syscall(SYS_gettid);

  if (ret < MMPA_ZERO) {
    return EN_ERROR;
  }

  return ret;
}

INT32 mmGetSystemTime(mmSystemTime_t *sys_time) {
  // Beijing olympics
  sys_time->wYear = 2008;
  sys_time->wMonth = 8;
  sys_time->wDay = 8;
  sys_time->wHour = 20;
  sys_time->wMinute = 8;
  sys_time->wSecond = 0;
  return 0;
}

INT32 mmAccess(const CHAR *path_name) {
  if (path_name == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = access(path_name, F_OK);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmStatGet(const CHAR *path, mmStat_t *buffer) {
  if ((path == NULL) || (buffer == NULL)) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = stat(path, buffer);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmGetFileSize(const CHAR *file_name, ULONGLONG *length) {
  if ((file_name == NULL) || (length == NULL)) {
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

INT32 mmAccess2(const CHAR *path_name, INT32 mode) {
  if (path_name == NULL) {
    return EN_INVALID_PARAM;
  }
  INT32 ret = access(path_name, mode);
  if (ret != EN_OK) {
    return EN_ERROR;
  }
  return EN_OK;
}

INT32 mmGetTimeOfDay(mmTimeval *time_val, mmTimezone *time_zone) {
  return 0;
}

INT32 mmGetErrorCode() {
  return errno;
}

CHAR *mmDlerror() {
  return dlerror();
}

VOID *mmDlopen(const CHAR *file_name, INT32 mode) {
  static int64_t tmp = 0;
  return &tmp;
}

INT32 mmDlclose(VOID *handle) {
  return 0;
}

VOID *mmDlsym(VOID *handle, const CHAR *func_name) {
  return nullptr;
}

INT32 mmGetPid() {
  return (INT32)getpid();
}

INT32 mmSetCurrentThreadName(const CHAR *name) {
  return EN_OK;
}

CHAR *mmDirName(CHAR *path) {
  if (path == NULL) {
    return NULL;
  }
  return dirname(path);
}

CHAR *mmBaseName(CHAR *path) {
  if (path == NULL) {
    return NULL;
  }
  return basename(path);
}

INT32 mmRWLockRDLock(mmRWLock_t *rw_lock) {
  if (rw_lock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_rdlock(rw_lock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockWRLock(mmRWLock_t *rw_lock) {
  if (rw_lock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_wrlock(rw_lock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRDLockUnLock(mmRWLock_t *rw_lock) {
  if (rw_lock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_unlock(rw_lock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmWRLockUnLock(mmRWLock_t *rw_lock) {
  if (rw_lock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_unlock(rw_lock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockInit(mmRWLock_t *rw_lock) {
  if (rw_lock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_init(rw_lock, NULL);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmRWLockDestroy(mmRWLock_t *rw_lock) {
  if (rw_lock == NULL) {
    return EN_INVALID_PARAM;
  }

  INT32 ret = pthread_rwlock_destroy(rw_lock);
  if (ret != MMPA_ZERO) {
    return EN_ERROR;
  }

  return EN_OK;
}

INT32 mmChmod(const CHAR *filename, int32_t mode) {
  if (filename == NULL) {
    return EN_INVALID_PARAM;
  }

  return chmod(filename, (UINT32)mode);
}

INT32 mmFStatGet(INT32 fd, mmStat_t *buf) {
  return fstat(fd, buf);
}

INT32 mmGetEnv(const CHAR *name, CHAR *value, UINT32 len) {
  const char *env = getenv(name);
  if (env == nullptr) {
    return EN_ERROR;
  }

  strncpy(value, env, len);
  return EN_OK;
}

INT32 mmSetEnv(const CHAR *name, const CHAR *value, INT32 overwrite) {
  if ((name == NULL) || (value == NULL)) {
    return EN_INVALID_PARAM;
  }

  return setenv(name, value, overwrite);
}

#ifdef __cplusplus
}
#endif
