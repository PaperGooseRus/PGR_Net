#pragma once

#include <cstddef>
#include <vector>

namespace pgr::neural {

/**
 * @brief Вещественное число вычислительного прототипа.
 *
 * Пока используем double как удобную численную опорную реализацию.
 * Это не бесконечно точное ℝ из Lean. Переход на float32 потребует
 * отдельной оценки погрешностей и повторной проверки, а не смены typedef.
 */
using Real = double;
using Vector = std::vector<Real>;

/**
 * @brief Небольшая неизменяемая плотная матрица, данные по строкам.
 *
 * Доступ по проверяемым индексам. Размер проверяется до rows*columns,
 * чтобы само вычисление размера не переполнило std::size_t.
 * Лимит относится к одной матрице, а не к памяти всего процесса.
 */
class Matrix final {
public:
    static constexpr std::size_t max_elements = 2'000'000;
    Matrix(std::size_t rows, std::size_t columns, Vector values);
    std::size_t rows() const noexcept;
    std::size_t columns() const noexcept;
    Real at(std::size_t row, std::size_t column) const;
private:
    std::size_t rows_;
    std::size_t columns_;
    Vector values_;
};

/** @brief Проверить, что вектор не содержит NaN и бесконечности. */
void require_finite(const Vector& values);

/** @brief Плотный слой y = W*x+b; размерности проверяются во время вызова. */
Vector affine(const Matrix& weights, const Vector& input, const Vector& bias);

/** @brief Поэлементный ReLU: max(0, x). Вход обязан быть конечным. */
Vector relu(const Vector& input);

/**
 * @brief Softmax с вычитанием максимума логитов.
 *
 * Не переполняет exp на больших положительных логитах, потому что её
 * аргумент <=0. Очень малые значения могут округлиться до нуля.
 * При непредставимом промежуточном значении возвращается исключение,
 * а не выдуманная категория или скрытый NaN.
 */
Vector softmax(const Vector& logits);

} // namespace pgr::neural
