#pragma once

#include <pgr/neural/math.hpp>
#include <cstddef>
#include <vector>

namespace pgr::neural {

/**
 * @brief Размеры сети. Значения >0; d+C+S ограничено для прототипа.
 *
 * Число компиляторов и стадий включает заранее предусмотренное значение
 * unknown. Соответствие индексов реальным именам пока задаёт вызывающая
 * сторона; строковый парсер и словарь токенов здесь не реализованы.
 */
struct Shape {
    std::size_t vocabulary;
    std::size_t embedding_dimension;
    std::size_t compiler_count;
    std::size_t stage_count;
    std::size_t hidden_size;
    std::size_t class_count;
    std::size_t max_tokens;
};

/** @brief Проверить скалярные пределы Shape до выделения буферов. */
void validate_shape(const Shape& shape);

/** @brief Пять массивов параметров сети, включая таблицу представлений. */
struct Parameters {
    Matrix embeddings; // V x d
    Matrix w1;         // H x (d + C + S)
    Vector b1;         // H
    Matrix w2;         // K x H
    Vector b2;         // K
};

/**
 * @brief Промежуточные результаты для обучения разработчика и тестирования.
 *
 * В промышленном режиме хранение всей трассы может быть необязательным.
 * Сейчас она мала и позволяет проверить каждый шаг без угадывания.
 */
struct ForwardTrace {
    Vector mean_embedding;
    Vector input;
    Vector hidden_before_relu;
    Vector hidden;
    Vector logits;
    Vector probabilities;
};

/**
 * @brief Неизменяемая сеть: CPU-прямой проход с переданными параметрами.
 *
 * Нет обучения, случайных весов, фоновых потоков, Qt, доступа к WinAPI
 * или выполнения команд. Вычисление происходит в потоке вызывающего кода.
 */
class Network final {
public:
    Network(Shape shape, Parameters parameters);
    ForwardTrace forward(const std::vector<std::size_t>& token_ids,
                         std::size_t compiler_index,
                         std::size_t stage_index) const;
    const Shape& shape() const noexcept;
    /** @brief Только чтение параметров для вычисления градиентов и аудита. */
    const Parameters& parameters() const noexcept;
private:
    Shape shape_;
    Parameters parameters_;
};

/** @brief Сеть с вручную заданными весами для численного примера. */
Network make_example_network();

} // namespace pgr::neural
