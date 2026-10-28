#include <pgr/neural/gradients.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace pgr::neural {
namespace {
Real checked(Real value) {
    if (!std::isfinite(value)) {
        throw std::overflow_error("Non-finite loss or gradient");
    } else {
        return value;
    }
}

Vector values(const Matrix& m) {
    Vector out;
    out.reserve(m.rows() * m.columns());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.columns(); ++j) {
            out.push_back(m.at(i, j));
        }
    }
    return out;
}

void accumulate(Vector& destination, const Vector& source) {
    for (std::size_t i = 0; i < destination.size(); ++i) {
        destination[i] = checked(destination[i] + source[i]);
    }
}

void divide(Vector& values, std::size_t count) {
    for (Real& value : values) {
        value = checked(value / static_cast<Real>(count));
    }
}
} // namespace

Real cross_entropy(const Vector& logits, std::size_t target) {
    if (logits.empty() || target >= logits.size()) {
        throw std::invalid_argument("Cross entropy requires valid logits and target");
    } else {
        require_finite(logits);
    }
    const Real maximum = *std::max_element(logits.begin(), logits.end());
    Real sum = 0.0;
    for (const Real value : logits) {
        sum = checked(sum + std::exp(checked(value - maximum)));
    }
    // Вычитаем максимум ДО exp; не берём log от уже округлённой вероятности.
    return checked(checked(maximum - logits[target]) + std::log(sum));
}

Real loss(const Network& network, const LabeledInput& sample) {
    return cross_entropy(network.forward(sample.tokens, sample.compiler_index,
                                         sample.stage_index).logits, sample.target_class);
}

LossAndGradients backward(const Network& network, const LabeledInput& sample) {
    const auto& s = network.shape();
    const auto& p = network.parameters();
    const auto trace = network.forward(sample.tokens, sample.compiler_index, sample.stage_index);
    const Real result_loss = cross_entropy(trace.logits, sample.target_class);
    const std::size_t input_size = trace.input.size();
    Vector delta = trace.probabilities;
    delta[sample.target_class] -= 1.0;
    Vector dw2(s.class_count * s.hidden_size, 0.0);
    Vector da(s.hidden_size, 0.0);
    for (std::size_t h = 0; h < s.hidden_size; ++h) {
        Real dh = 0.0;
        for (std::size_t k = 0; k < s.class_count; ++k) {
            dw2[k * s.hidden_size + h] = checked(delta[k] * trace.hidden[h]);
            dh = checked(dh + checked(p.w2.at(k, h) * delta[k]));
        }
        if (trace.hidden_before_relu[h] > 0.0) {
            da[h] = dh;
        } else {
            // Включая ровно 0: это выбранная конвенция субградиента ReLU.
            da[h] = 0.0;
        }
    }
    Vector dw1(s.hidden_size * input_size, 0.0);
    Vector dx(input_size, 0.0);
    for (std::size_t h = 0; h < s.hidden_size; ++h) {
        for (std::size_t j = 0; j < input_size; ++j) {
            dw1[h * input_size + j] = checked(da[h] * trace.input[j]);
            dx[j] = checked(dx[j] + checked(p.w1.at(h, j) * da[h]));
        }
    }
    Vector de(s.vocabulary * s.embedding_dimension, 0.0);
    for (const auto token : sample.tokens) {
        for (std::size_t d = 0; d < s.embedding_dimension; ++d) {
            const std::size_t index = token * s.embedding_dimension + d;
            de[index] = checked(de[index] + dx[d] / static_cast<Real>(sample.tokens.size()));
        }
    }
    return LossAndGradients{result_loss, Gradients{
        Matrix(s.vocabulary, s.embedding_dimension, std::move(de)),
        Matrix(s.hidden_size, input_size, std::move(dw1)), std::move(da),
        Matrix(s.class_count, s.hidden_size, std::move(dw2)), std::move(delta)}};
}

LossAndGradients backward_batch(const Network& network, const std::vector<LabeledInput>& batch) {
    if (batch.empty() || batch.size() > 256) {
        throw std::invalid_argument("Batch size must be in [1,256]");
    } else {
        // Один результат обрабатывается за раз, вычислительный граф не хранится.
    }
    const auto& s = network.shape();
    const auto input_size = s.embedding_dimension + s.compiler_count + s.stage_count;
    Vector de(s.vocabulary*s.embedding_dimension,0.0), dw1(s.hidden_size*input_size,0.0);
    Vector db1(s.hidden_size,0.0), dw2(s.class_count*s.hidden_size,0.0), db2(s.class_count,0.0);
    Real total = 0.0;
    for (const auto& sample : batch) {
        const auto item = backward(network,sample);
        total = checked(total + item.loss);
        accumulate(de,values(item.gradients.embeddings));
        accumulate(dw1,values(item.gradients.w1));
        accumulate(db1,item.gradients.b1);
        accumulate(dw2,values(item.gradients.w2));
        accumulate(db2,item.gradients.b2);
    }
    divide(de,batch.size()); divide(dw1,batch.size()); divide(db1,batch.size());
    divide(dw2,batch.size()); divide(db2,batch.size());
    return LossAndGradients{checked(total/static_cast<Real>(batch.size())), Gradients{
        Matrix(s.vocabulary,s.embedding_dimension,std::move(de)),
        Matrix(s.hidden_size,input_size,std::move(dw1)),std::move(db1),
        Matrix(s.class_count,s.hidden_size,std::move(dw2)),std::move(db2)}};
}
} // namespace pgr::neural
