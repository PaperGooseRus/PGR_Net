#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace pgr::neural::detail {
/**
 * @brief Полностью определённый unsigned-генератор для воспроизводимых опытов.
 *
 * Переполнение uint64_t — определённая арифметика по модулю 2^64.
 * Не используем uniform_real_distribution: его конкретный алгоритм может
 * отличаться между стандартными библиотеками MSVC/MinGW/Linux.
 */
class Random final {
public:
    explicit Random(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        state_ += UINT64_C(0x9e3779b97f4a7c15);
        std::uint64_t z = state_;
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    double unit() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    std::size_t bounded(std::size_t bound) {
        if (bound == 0) {
            throw std::invalid_argument("Random bound must be positive");
        } else {
            // Rejection sampling убирает перекос от остатка деления.
        }
        const auto b = static_cast<std::uint64_t>(bound);
        const std::uint64_t threshold = (std::uint64_t{0} - b) % b;
        for (;;) {
            const auto value = next();
            if (value >= threshold) {
                return static_cast<std::size_t>(value % b);
            } else {
                // Отброшенное значение не влияет на выбор индекса.
            }
        }
    }
private:
    std::uint64_t state_;
};
} // namespace pgr::neural::detail
