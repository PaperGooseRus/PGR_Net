#pragma once
#include <pgr/neural/network.hpp>

namespace pgr::neural {

/** @brief Данные одного размеченного примера; строкового токенизатора пока нет. */
struct LabeledInput {
    std::vector<std::size_t> tokens;
    std::size_t compiler_index;
    std::size_t stage_index;
    std::size_t target_class;
};

/** @brief Производные по тем же пяти массивам, что и Parameters. */
struct Gradients {
    Matrix embeddings;
    Matrix w1;
    Vector b1;
    Matrix w2;
    Vector b2;
};

struct LossAndGradients {
    Real loss;
    Gradients gradients;
};

/**
 * @brief L=log(sum exp(z-max))+max-z[target], без вычисления log(p[target]).
 *
 * Сохраняет конечную потерю при underflow целевой вероятности до 0.
 * Возвращает исключение при неверном индексе или непредставимой арифметике.
 */
Real cross_entropy(const Vector& logits, std::size_t target);

/** @brief Только значение потери одного примера. Не обновляет веса. */
Real loss(const Network& network, const LabeledInput& sample);

/**
 * @brief Аналитический градиент одного примера.
 *
 * dL/dz = softmax(z)-one_hot(target), ReLU'(0) условно принимаем равным 0.
 * Повторения токена суммируют вклады в одну строку таблицы embeddings.
 * Никакого изменения Network или запуска оптимизатора здесь нет.
 */
LossAndGradients backward(const Network& network, const LabeledInput& sample);

/**
 * @brief Средняя потеря и средний градиент батча; примеры обрабатываются последовательно.
 *
 * Пустой батч запрещён; защитный лимит 256 примеров. Этот лимит не является
 * ограничением памяти вызывающего кода или квотой CPU/ОС.
 */
LossAndGradients backward_batch(const Network& network, const std::vector<LabeledInput>& batch);

} // namespace pgr::neural
