#include "file_io.hpp"
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace pgr::neural::detail {
namespace {
// Счётчик исключает совпадения имён между потоками процесса. Межпроцессную
// коллизию закрывает CREATE_NEW/O_EXCL, а не предполагаемая уникальность часов.
std::atomic<std::uint64_t> sequence{0};
std::filesystem::path temp_name(const std::filesystem::path& parent) {
    const auto clock = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return parent / (".pgrnn-" + std::to_string(clock) + "-" +
        std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ".tmp");
}

#if defined(_WIN32)
[[noreturn]] void system_failure(const char* action, DWORD code) {
    throw std::system_error(static_cast<int>(code), std::system_category(), action);
}
class TempFile final {
public:
    explicit TempFile(const std::filesystem::path& parent) {
        for (unsigned attempt = 0; attempt < 64; ++attempt) {
            path_ = temp_name(parent);
            handle_ = CreateFileW(path_.c_str(), GENERIC_WRITE, 0, nullptr,
                                  CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle_ != INVALID_HANDLE_VALUE) {
                owned_ = true;
                return;
            } else {
                const auto error = GetLastError();
                if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS) {
                    system_failure("Create PGRNN temporary file", error);
                } else {
                    // Чужой файл не удаляем и не перезаписываем; пробуем имя.
                }
            }
        }
        throw std::runtime_error("Cannot reserve PGRNN temporary name");
    }
    ~TempFile() {
        if (handle_ != INVALID_HANDLE_VALUE) { CloseHandle(handle_); }
        else { /* Файл уже закрыт перед публикацией. */ }
        if (owned_) { DeleteFileW(path_.c_str()); }
        else { /* Имя перенесено; удалять цель нельзя. */ }
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    void write_and_close(const ModelBytes& bytes) {
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            // Файл <=64 MiB: длина гарантированно помещается в DWORD.
            DWORD written = 0;
            if (!WriteFile(handle_, bytes.data() + offset,
                           static_cast<DWORD>(bytes.size() - offset), &written, nullptr)) {
                system_failure("Write PGRNN temporary file", GetLastError());
            } else if (written == 0) {
                throw std::runtime_error("PGRNN zero-length write");
            } else {
                offset += written;
            }
        }
        if (!FlushFileBuffers(handle_)) {
            system_failure("Flush PGRNN temporary file", GetLastError());
        } else {
            // Ошибка flush не превращается в успешное сохранение.
        }
        const auto handle = handle_;
        handle_ = INVALID_HANDLE_VALUE;
        if (!CloseHandle(handle)) {
            system_failure("Close PGRNN temporary file", GetLastError());
        } else {
            // Полностью записанный закрытый файл готов к публикации.
        }
    }
    void publish(const std::filesystem::path& target) {
        // Нет MOVEFILE_REPLACE_EXISTING и COPY_ALLOWED. Цель должна быть
        // новой, перенос — внутри одного каталога/тома, без окна перезаписи.
        if (!MoveFileExW(path_.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH)) {
            system_failure("Publish PGRNN without overwrite", GetLastError());
        } else {
            owned_ = false;
        }
    }
private:
    std::filesystem::path path_;
    HANDLE handle_ = INVALID_HANDLE_VALUE;
    bool owned_ = false;
};
#else
[[noreturn]] void system_failure(const char* action, int code) {
    throw std::system_error(code, std::generic_category(), action);
}
class TempFile final {
public:
    explicit TempFile(const std::filesystem::path& parent) {
        for (unsigned attempt = 0; attempt < 64; ++attempt) {
            path_ = temp_name(parent);
            fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
            if (fd_ >= 0) {
                owned_ = true;
                return;
            } else if (errno != EEXIST) {
                system_failure("Create PGRNN temporary file", errno);
            } else {
                // O_EXCL также не следует существующей символьной ссылке.
            }
        }
        throw std::runtime_error("Cannot reserve PGRNN temporary name");
    }
    ~TempFile() {
        if (fd_ >= 0) { ::close(fd_); }
        else { /* Дескриптор уже закрыт. */ }
        if (owned_) { ::unlink(path_.c_str()); }
        else { /* Временное имя уже удалено. */ }
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    void write_and_close(const ModelBytes& bytes) {
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto written = ::write(fd_, bytes.data() + offset, bytes.size() - offset);
            if (written < 0 && errno == EINTR) {
                continue;
            } else if (written < 0) {
                system_failure("Write PGRNN temporary file", errno);
            } else if (written == 0) {
                throw std::runtime_error("PGRNN zero-length write");
            } else {
                offset += static_cast<std::size_t>(written);
            }
        }
        if (::fsync(fd_) != 0) {
            system_failure("Flush PGRNN temporary file", errno);
        } else {
            // Публикуем только после успешного завершения записи.
        }
        const auto fd = fd_;
        fd_ = -1;
        if (::close(fd) != 0) {
            // close не повторяем: номер мог быть уже переиспользован ОС.
            system_failure("Close PGRNN temporary file", errno);
        } else {
            // Полная запись завершена.
        }
    }
    void publish(const std::filesystem::path& target) {
        // rename на POSIX может заменить цель. link создаёт НОВОЕ имя целого
        // закрытого файла или отказывает, если имя занято. ФС должна уметь
        // hard links; небезопасного fallback с перезаписью здесь нет.
        if (::link(path_.c_str(), target.c_str()) != 0) {
            system_failure("Publish PGRNN without overwrite", errno);
        } else if (::unlink(path_.c_str()) == 0) {
            owned_ = false;
        } else {
            // Цель уже опубликована успешно. Деструктор ещё раз попытается
            // убрать служебную ссылку; рабочую цель он никогда не удаляет.
        }
    }
private:
    std::filesystem::path path_;
    int fd_ = -1;
    bool owned_ = false;
};
#endif
} // namespace

void publish_new_file(const std::filesystem::path& file, const ModelBytes& bytes) {
    if (file.empty() || file.filename().empty()) {
        throw std::invalid_argument("PGRNN output filename is empty");
    } else {
        // Проверка наличия цели здесь была бы лишь предварительной:
        // окончательное решение принимает атомарная операция публикации ОС.
    }
    const auto target = std::filesystem::absolute(file);
    const auto parent = target.parent_path();
    if (!std::filesystem::is_directory(parent)) {
        throw std::invalid_argument("PGRNN parent directory must exist");
    } else {
        TempFile temp(parent);
        temp.write_and_close(bytes);
        temp.publish(target);
    }
}
} // namespace pgr::neural::detail
