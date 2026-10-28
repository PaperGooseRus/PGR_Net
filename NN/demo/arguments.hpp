#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace pgr::neural::cli {

/**
 * @brief Аргументы CLI в UTF-8, включая Windows с кириллицей в пути.
 *
 * Обычный char** argv Windows может потерять символы до вызова main.
 * Поэтому на Windows берём широкую командную строку ОС, затем явно
 * преобразуем UTF-16 -> UTF-8. Файловые пути далее создаём через u8path.
 */
inline std::vector<std::string> arguments(int argc, char** argv) {
#if defined(_WIN32)
    static_cast<void>(argc);
    static_cast<void>(argv);
    int count = 0;
    struct WideArguments {
        wchar_t** value;
        ~WideArguments() {
            if (value != nullptr) { LocalFree(value); }
            else { /* Выделения не было. */ }
        }
    } wide{CommandLineToArgvW(GetCommandLineW(), &count)};
    if (wide.value == nullptr) {
        throw std::runtime_error("Cannot read Windows command-line arguments");
    } else {
        std::vector<std::string> result;
        for (int i = 0; i < count; ++i) {
            const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                wide.value[i], -1, nullptr, 0, nullptr, nullptr);
            if (bytes <= 0) {
                throw std::runtime_error("Cannot encode Windows argument as UTF-8");
            } else {
                std::string text(static_cast<std::size_t>(bytes), '\0');
                if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.value[i], -1,
                    text.data(), bytes, nullptr, nullptr) != bytes) {
                    throw std::runtime_error("Windows argument conversion failed");
                } else {
                    text.pop_back();
                    result.push_back(std::move(text));
                }
            }
        }
        return result;
    }
#else
    return {argv, argv + argc};
#endif
}
inline std::uint64_t integer(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Expected an unsigned decimal integer");
    } else {
        return std::stoull(text);
    }
}
inline std::size_t size_value(const std::string& text) {
    const auto value = integer(text);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::out_of_range("Integer does not fit size_t");
    } else {
        return static_cast<std::size_t>(value);
    }
}
} // namespace pgr::neural::cli
