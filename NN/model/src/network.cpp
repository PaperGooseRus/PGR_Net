#include <pgr/neural/network.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace pgr::neural {
namespace {
constexpr std::size_t max_dimension = 4096;

void check_shape(const Shape& shape) {
    // Проверки ограничивают внутренние буферы, но не являются квотой RAM ОС.
    const std::size_t values[] = {shape.embedding_dimension, shape.compiler_count,
        shape.stage_count, shape.hidden_size, shape.class_count, shape.max_tokens};
    for (const std::size_t value : values) {
        if (value == 0 || value > max_dimension) {
            throw std::invalid_argument("Shape dimension is outside prototype limits");
        } else {
            // Каждый размер мал, сложение d+C+S теперь не переполняется.
        }
    }
    if (shape.vocabulary == 0 || shape.vocabulary > Matrix::max_elements) {
        throw std::invalid_argument("Invalid vocabulary size");
    } else {
        // Лимит всей таблицы дополнительно проверен в конструкторе Matrix.
    }
}

bool has_shape(const Matrix& matrix, std::size_t rows, std::size_t columns) {
    return matrix.rows() == rows && matrix.columns() == columns;
}
} // namespace

void validate_shape(const Shape& shape) { check_shape(shape); }

Network::Network(Shape shape, Parameters parameters)
    : shape_(shape), parameters_(std::move(parameters)) {
    validate_shape(shape_);
    const std::size_t input_size = shape_.embedding_dimension + shape_.compiler_count + shape_.stage_count;
    if (!has_shape(parameters_.embeddings, shape_.vocabulary, shape_.embedding_dimension) ||
        !has_shape(parameters_.w1, shape_.hidden_size, input_size) ||
        !has_shape(parameters_.w2, shape_.class_count, shape_.hidden_size) ||
        parameters_.b1.size() != shape_.hidden_size ||
        parameters_.b2.size() != shape_.class_count) {
        throw std::invalid_argument("Network parameter dimensions do not match Shape");
    } else {
        require_finite(parameters_.b1);
        require_finite(parameters_.b2);
    }
}

const Shape& Network::shape() const noexcept { return shape_; }
const Parameters& Network::parameters() const noexcept { return parameters_; }

ForwardTrace Network::forward(const std::vector<std::size_t>& token_ids,
                              std::size_t compiler_index,
                              std::size_t stage_index) const {
    if (token_ids.empty()) {
        throw std::invalid_argument("Token sequence must not be empty");
    } else if (token_ids.size() > shape_.max_tokens) {
        throw std::length_error("Token sequence exceeds max_tokens");
    } else if (compiler_index >= shape_.compiler_count || stage_index >= shape_.stage_count) {
        throw std::out_of_range("Context index is out of range");
    } else {
        // Сначала проверяем все индексы, потом начинаем вычисление.
    }
    for (const std::size_t token : token_ids) {
        if (token >= shape_.vocabulary) {
            throw std::out_of_range("Token index is outside the vocabulary");
        } else {
            // Для неизвестного слова вызывающий код обязан передать индекс <unk>.
        }
    }
    ForwardTrace trace;
    trace.mean_embedding.assign(shape_.embedding_dimension, 0.0);
    for (const std::size_t token : token_ids) {
        for (std::size_t d = 0; d < shape_.embedding_dimension; ++d) {
            trace.mean_embedding[d] += parameters_.embeddings.at(token, d);
            if (!std::isfinite(trace.mean_embedding[d])) {
                throw std::overflow_error("Embedding sum overflow");
            } else {
                // Переполнение не маскируется последующим делением на n.
            }
        }
    }
    for (Real& value : trace.mean_embedding) {
        value /= static_cast<Real>(token_ids.size());
    }
    trace.input = trace.mean_embedding;
    const std::size_t compiler_offset = trace.input.size();
    trace.input.resize(compiler_offset + shape_.compiler_count + shape_.stage_count, 0.0);
    trace.input[compiler_offset + compiler_index] = 1.0;
    trace.input[compiler_offset + shape_.compiler_count + stage_index] = 1.0;
    trace.hidden_before_relu = affine(parameters_.w1, trace.input, parameters_.b1);
    trace.hidden = relu(trace.hidden_before_relu);
    trace.logits = affine(parameters_.w2, trace.hidden, parameters_.b2);
    trace.probabilities = softmax(trace.logits);
    return trace;
}

} // namespace pgr::neural
