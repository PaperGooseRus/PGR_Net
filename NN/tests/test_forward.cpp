#include <pgr/neural/network.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace pgr::neural;

/** @brief Проверки не зависят от assert, поэтому работают и в Release. */
void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    } else {
        // Утверждение выполнено.
    }
}

void near(Real actual, Real expected, Real tolerance = 1e-12) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          "Numeric mismatch: actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

void vector_near(const Vector& actual, const Vector& expected) {
    check(actual.size() == expected.size(), "Vector size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        near(actual[i], expected[i]);
    }
}

template<class Exception, class Callable>
void throws(Callable action) {
    bool caught = false;
    try {
        action();
    } catch (const Exception&) {
        caught = true;
    }
    check(caught, "Expected exception not thrown");
}

void manual_forward() {
    // Полный ручной расчёт приведён в docs/MANUAL_EXAMPLE_RU.md.
    // Ожидания заданы независимо от функций affine/relu/softmax.
    const auto result = make_example_network().forward({1, 2}, 1, 0);
    vector_near(result.mean_embedding, {1, 1});
    vector_near(result.input, {1, 1, 0, 1, 1, 0});
    vector_near(result.hidden_before_relu, {2, -1});
    vector_near(result.hidden, {2, 0});
    vector_near(result.logits, {2, 0});
    vector_near(result.probabilities, {0.8807970779778824, 0.1192029220221176});
}

void context_changes_result() {
    const auto result = make_example_network().forward({1, 2}, 1, 1);
    vector_near(result.input, {1, 1, 0, 1, 0, 1});
    vector_near(result.hidden_before_relu, {1, -1});
    vector_near(result.probabilities, {0.7310585786300049, 0.2689414213699951});
}

void average_not_sum() {
    const auto network = make_example_network();
    vector_near(network.forward({1}, 1, 0).mean_embedding, {2, 0});
    vector_near(network.forward({1, 1, 1}, 1, 0).mean_embedding, {2, 0});
}

void permutation_invariance() {
    const auto network = make_example_network();
    // Это одновременно проверка реализации и демонстрация ограничения:
    // средний вектор не умеет различать последовательности переставленных слов.
    vector_near(network.forward({1, 2}, 1, 0).probabilities,
                network.forward({2, 1}, 1, 0).probabilities);
}

void affine_reference() {
    const Matrix w(2, 3, {1, 2, 3, -1, 0, 1});
    vector_near(affine(w, {1, 2, 3}, {1, -1}), {15, 1});
}

void relu_boundaries() {
    vector_near(relu({-3, 0, 2}), {0, 0, 2});
}

void softmax_stability() {
    const auto values = softmax({1001, 1000, -1000});
    vector_near(values, {0.7310585786300049, 0.2689414213699951, 0.0});
    near(std::accumulate(values.begin(), values.end(), 0.0), 1.0);
    for (const Real value : values) {
        check(value >= 0 && value <= 1, "Softmax value outside [0,1]");
    }
}

void softmax_common_shift() {
    vector_near(softmax({2, 0}), softmax({1002, 1000}));
    vector_near(softmax({0}), {1});
}

void seven_outputs_supported() {
    const Network network(Shape{1, 1, 1, 1, 1, 7, 8}, Parameters{
        Matrix(1, 1, {0}), Matrix(1, 3, {0, 0, 0}), {0},
        Matrix(7, 1, Vector(7, 0.0)), Vector(7, 0.0)});
    vector_near(network.forward({0}, 0, 0).probabilities, Vector(7, 1.0 / 7.0));
}

void invalid_matrix_data() {
    throws<std::invalid_argument>([] { Matrix matrix(0, 2, {}); });
    throws<std::invalid_argument>([] { Matrix matrix(2, 2, {1}); });
    throws<std::length_error>([] { Matrix matrix(std::numeric_limits<std::size_t>::max(), 2, {}); });
    const Matrix matrix(1, 1, {1});
    throws<std::out_of_range>([&] { static_cast<void>(matrix.at(1, 0)); });
}

void invalid_network_parameters() {
    throws<std::invalid_argument>([] {
        Network network(Shape{3, 2, 2, 2, 2, 2, 128}, Parameters{
            Matrix(3, 2, Vector(6, 0)), Matrix(2, 5, Vector(10, 0)), {0, 0},
            Matrix(2, 2, Vector(4, 0)), {0, 0}});
    });
    throws<std::invalid_argument>([] {
        Network network(Shape{1, 1, 1, 1, 1, 0, 8}, Parameters{
            Matrix(1, 1, {0}), Matrix(1, 3, {0, 0, 0}), {0},
            Matrix(1, 1, {0}), {0}});
    });
}

void invalid_sequence_and_context() {
    const auto network = make_example_network();
    throws<std::invalid_argument>([&] { static_cast<void>(network.forward({}, 1, 0)); });
    throws<std::out_of_range>([&] { static_cast<void>(network.forward({3}, 1, 0)); });
    throws<std::out_of_range>([&] { static_cast<void>(network.forward({1}, 2, 0)); });
    throws<std::out_of_range>([&] { static_cast<void>(network.forward({1}, 1, 2)); });
    throws<std::length_error>([&] { static_cast<void>(network.forward(std::vector<std::size_t>(129, 0), 1, 0)); });
}

void invalid_floating_point_inputs() {
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();
    throws<std::invalid_argument>([&] { static_cast<void>(softmax({nan, 0})); });
    throws<std::invalid_argument>([&] { static_cast<void>(softmax({inf, 0})); });
    throws<std::invalid_argument>([&] { static_cast<void>(relu({nan})); });
    throws<std::invalid_argument>([&] { Matrix matrix(1, 1, {inf}); });
    throws<std::invalid_argument>([] { static_cast<void>(softmax({})); });
}

void arithmetic_overflow_is_visible() {
    const Real largest = std::numeric_limits<Real>::max();
    throws<std::overflow_error>([&] { static_cast<void>(affine(Matrix(1, 1, {largest}), {2}, {0})); });
    // Это отдельный выход за численную область нашего double-прототипа.
    throws<std::overflow_error>([&] { static_cast<void>(softmax({largest, -largest})); });
}

void affine_dimension_error() {
    throws<std::invalid_argument>([] { static_cast<void>(affine(Matrix(1, 2, {1, 1}), {1}, {0})); });
    throws<std::invalid_argument>([] { static_cast<void>(affine(Matrix(1, 2, {1, 1}), {1, 1}, {})); });
}
} // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"manual_forward", manual_forward}, {"context_changes_result", context_changes_result},
        {"average_not_sum", average_not_sum}, {"permutation_invariance", permutation_invariance},
        {"affine_reference", affine_reference}, {"relu_boundaries", relu_boundaries},
        {"softmax_stability", softmax_stability}, {"softmax_common_shift", softmax_common_shift},
        {"seven_outputs_supported", seven_outputs_supported}, {"invalid_matrix_data", invalid_matrix_data},
        {"invalid_network_parameters", invalid_network_parameters},
        {"invalid_sequence_and_context", invalid_sequence_and_context},
        {"invalid_floating_point_inputs", invalid_floating_point_inputs},
        {"arithmetic_overflow_is_visible", arithmetic_overflow_is_visible},
        {"affine_dimension_error", affine_dimension_error}
    };
    std::size_t failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << "Tests: " << tests.size() << "; failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
