#pragma once
#include <pgr/neural/gradients.hpp>
#include <cstdint>

namespace pgr::neural {

/**
 * @brief Новая сеть с детерминированной инициализацией.
 *
 * E: uniform[-0.5,0.5); W1/W2: Xavier uniform; b1=0.1; b2=0.
 * Seed задаёт битовый поток SplitMix64. Это не криптографический генератор.
 */
Network initialize_network(Shape shape, std::uint64_t seed);

/**
 * @brief Чистый шаг SGD: новая Network с theta-lr*gradient.
 *
 * Проверяет все размерности и конечность. Исходная сеть не меняется даже при
 * исключении. Momentum, Adam, weight decay и clipping пока отсутствуют.
 */
Network sgd_step(const Network& network, const Gradients& gradient, Real learning_rate);

struct TrainOptions {
    std::size_t epochs = 80;
    std::size_t batch_size = 4;
    Real learning_rate = 0.15;
    std::uint64_t shuffle_seed = 20261028;
    bool shuffle = true;
};

/** @brief Метрики на том же учебном наборе; НЕ независимая тестовая точность. */
struct EpochMetrics {
    std::size_t epoch;
    std::size_t updates;
    Real training_loss;
    Real training_accuracy;
};

struct TrainResult {
    Network network;
    std::vector<EpochMetrics> history; // Включает epoch=0 до первого обновления.
    std::size_t updates;
};

/**
 * @brief Ограниченный учебный mini-batch SGD.
 *
 * Инициализация вызывается отдельно, параметры переданной сети не сбрасываются.
 * Seed здесь влияет только на порядок. До запуска проверяются все примеры.
 * Один поток; не более 100000 обновлений, 10000 эпох, 100000 примеров,
 * размер батча 1..256. Лимиты не являются квотой CPU/RAM всей системы.
 */
TrainResult train_sgd(Network network, const std::vector<LabeledInput>& dataset,
                      const TrainOptions& options);

/** @brief Маленькая явно учебная задача: токены 1/2 против токенов 3/4. */
std::vector<LabeledInput> make_toy_dataset();
Shape toy_shape();

} // namespace pgr::neural
