// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cwchar>

#include <windows.h>

#include <dbghelp.h>

#include "common/logging.h"
#include "common/windows/crash_handler.h"

#define LOG_CRASH(...) ::Common::Log::FmtCrashMessage(__FILE__, __LINE__, __func__, __VA_ARGS__)

namespace Common::Windows {
namespace {

using MiniDumpWriteDumpFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE,
                                          PMINIDUMP_EXCEPTION_INFORMATION,
                                          PMINIDUMP_USER_STREAM_INFORMATION,
                                          PMINIDUMP_CALLBACK_INFORMATION);
using GetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);

// Everything the handler needs is resolved at install: loading a DLL from a crashed process can
// deadlock on the loader lock.
MiniDumpWriteDumpFn write_dump{};
GetThreadDescriptionFn get_thread_description{};
LPTOP_LEVEL_EXCEPTION_FILTER previous_filter{};
wchar_t dump_path[MAX_PATH]{};
std::atomic<DWORD> reporting_thread{};

struct CodeAddress {
    char module[64];
    unsigned long long offset;
};

CodeAddress Locate(const void* address) {
    CodeAddress result{"?", reinterpret_cast<uintptr_t>(address)};
    HMODULE module{};
    char path[MAX_PATH];
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCSTR>(address), &module) ||
        GetModuleFileNameA(module, path, sizeof(path)) == 0) {
        return result;
    }
    const char* slash = std::strrchr(path, '\\');
    std::snprintf(result.module, sizeof(result.module), "%s", slash ? slash + 1 : path);
    result.offset -= reinterpret_cast<uintptr_t>(module);
    return result;
}

void Report(EXCEPTION_POINTERS* pointers) {
    // Called on the crashed thread's stack, which the dispatcher's frames sit on top of: the
    // stack from the faulting frame on is the part after the exception address.
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    const void* fault = pointers ? pointers->ExceptionRecord->ExceptionAddress : nullptr;
    const USHORT first =
        pointers ? static_cast<USHORT>(std::find(frames, frames + count, fault) - frames) : 0;
    char stack[1024] = "";
    std::size_t used = 0;
    for (USHORT i = first < count ? first : 0; i < count && frames[i]; ++i) {
        const CodeAddress frame = Locate(frames[i]);
        const int written = std::snprintf(stack + used, sizeof(stack) - used, " %s+%#llx",
                                          frame.module, frame.offset);
        if (written < 0 || used + static_cast<std::size_t>(written) >= sizeof(stack)) {
            break;
        }
        used += static_cast<std::size_t>(written);
    }

    char thread_name[64] = "";
    PWSTR description{};
    if (get_thread_description &&
        SUCCEEDED(get_thread_description(GetCurrentThread(), &description))) {
        WideCharToMultiByte(CP_UTF8, 0, description, -1, thread_name, sizeof(thread_name) - 1,
                            nullptr, nullptr);
        LocalFree(description);
    }

    if (pointers) {
        const EXCEPTION_RECORD& record = *pointers->ExceptionRecord;
        const CodeAddress at = Locate(fault);
        LOG_CRASH("Host crash: unhandled exception {:#010x} at {}+{:#x} on thread {} \"{}\", "
                  "info {:#x} {:#x}; stack:{}",
                  record.ExceptionCode, at.module, at.offset, GetCurrentThreadId(), thread_name,
                  record.NumberParameters > 0 ? record.ExceptionInformation[0] : 0,
                  record.NumberParameters > 1 ? record.ExceptionInformation[1] : 0, stack);
    } else {
        LOG_CRASH("Host crash: abort() on thread {} \"{}\"; stack:{}", GetCurrentThreadId(),
                  thread_name, stack);
    }

    if (!write_dump) {
        return;
    }
    const HANDLE file = CreateFileW(dump_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return;
    }
    MINIDUMP_EXCEPTION_INFORMATION information{GetCurrentThreadId(), pointers, FALSE};
    const BOOL written =
        write_dump(GetCurrentProcess(), GetCurrentProcessId(), file, MiniDumpWithThreadInfo,
                   pointers ? &information : nullptr, nullptr, nullptr);
    const DWORD error = written ? 0 : GetLastError();
    CloseHandle(file);
    LOG_CRASH("Host crash: minidump of process {} {} ({:#x})", GetCurrentProcessId(),
              written ? "written next to the log as eden_crash.dmp" : "failed", error);
}

// Only the first thread to crash reports. A second one waits a while for it to end the process,
// then falls through to the previous handling, so a report stuck on a lock the second thread
// holds cannot hang the process. A crash inside the report falls through at once.
bool BeginReport() {
    DWORD expected = 0;
    const DWORD self = GetCurrentThreadId();
    if (reporting_thread.compare_exchange_strong(expected, self)) {
        return true;
    }
    if (expected != self) {
        Sleep(10000);
    }
    return false;
}

LONG WINAPI OnUnhandledException(EXCEPTION_POINTERS* pointers) {
    if (BeginReport()) {
        Report(pointers);
    }
    return previous_filter ? previous_filter(pointers) : EXCEPTION_CONTINUE_SEARCH;
}

// std::terminate (an uncaught C++ exception, also on a noexcept thread entry) and failed fatal
// asserts end in abort(), which the C runtime turns into a fail-fast that skips the filter above.
void OnAbort(int) {
    if (BeginReport()) {
        Report(nullptr);
    }
}

} // Anonymous namespace

void InstallCrashHandler(const std::filesystem::path& dump_directory) {
    const std::wstring path = (dump_directory / "eden_crash.dmp").wstring();
    if (path.size() < MAX_PATH) {
        std::wmemcpy(dump_path, path.c_str(), path.size() + 1);
        if (const HMODULE dbghelp = LoadLibraryW(L"dbghelp.dll")) {
            write_dump = reinterpret_cast<MiniDumpWriteDumpFn>(
                reinterpret_cast<void*>(GetProcAddress(dbghelp, "MiniDumpWriteDump")));
        }
    }
    get_thread_description = reinterpret_cast<GetThreadDescriptionFn>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"KernelBase.dll"), "GetThreadDescription")));
    previous_filter = SetUnhandledExceptionFilter(OnUnhandledException);
    std::signal(SIGABRT, OnAbort);
}

} // namespace Common::Windows
