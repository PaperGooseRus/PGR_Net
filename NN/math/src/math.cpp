#include <pgr/neural/math.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace pgr::neural {

namespace {
/** @brief Проверка результата арифметики, не входных весов. */
void require_finite_result(Real value) {
    if (!std::isfinite(value)) {
        throw std::overflow_error("Non-finite intermediate arithmetic result");
    } else {
        // Конечное значение пригодно для следующей операции.
    }
}
} // namespace

void require_finite(const Vector& values) {
    for (const Real value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Input contains NaN or infinity");
        } else {
            // Продолжаем проверку остальных компонентов.
        }
    }
}

Matrix::Matrix(std::size_t rows, std::size_t columns, Vector values)
    : rows_(rows), columns_(columns), values_(std::move(values)) {
    if (rows_ == 0 || columns_ == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    } else if (rows_ > max_elements / columns_) {
        throw std::length_error("Matrix exceeds configured element limit");
    } else if (values_.size() != rows_ * columns_) {
        throw std::invalid_argument("Matrix data size does not match dimensions");
    } else {
        require_finite(values_);
    }
}

std::size_t Matrix::rows() const noexcept { return rows_; }
std::size_t Matrix::columns() const noexcept { return columns_; }

Real Matrix::at(std::size_t row, std::size_t column) const {
    if (row >= rows_ || column >= columns_) {
        throw std::out_of_range("Matrix index is out of range");
    } else {
        return values_[row * columns_ + column];
    }
}

Vector affine(const Matrix& weights, const Vector& input, const Vector& bias) {
    if (input.size() != weights.columns() || bias.size() != weights.rows()) {
        throw std::invalid_argument("Affine layer dimension mismatch");
    } else {
        require_finite(input);
        require_finite(bias);
    }
    Vector output(weights.rows(), 0.0);
    for (std::size_t row = 0; row < weights.rows(); ++row) {
        Real sum = bias[row];
        for (std::size_t column = 0; column < weights.columns(); ++column) {
            const Real product = weights.at(row, column) * input[column];
            require_finite_result(product);
            sum += product;
            require_finite_result(sum);
        }
        output[row] = sum;
    }
    return output;
}

Vector relu(const Vector& input) {
    require_finite(input);
    Vector output = input;
    for (Real& value : output) {
        value = std::max(0.0, value);
    }
    return output;
}

Vector softmax(const Vector& logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Softmax requires at least one logit");
    } else {
        require_finite(logits);
    }
    const Real maximum = *std::max_element(logits.begin(), logits.end());
    Vector output(logits.size(), 0.0);
    Real denominator = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const Real shifted = logits[i] - maximum;
        require_finite_result(shifted);
        output[i] = std::exp(shifted);
        require_finite_result(output[i]);
        denominator += output[i];
        require_finite_result(denominator);
    }
    if (!(denominator > 0.0)) {
        throw std::overflow_error("Invalid softmax denominator");
    } else {
        // Один элемент exp(max-max)=1; для конечных логитов знаменатель >0.
    }
    for (Real& value : output) {
        value /= denominator;
        require_finite_result(value);
    }
    return output;
}

} // namespace pgr::neural
