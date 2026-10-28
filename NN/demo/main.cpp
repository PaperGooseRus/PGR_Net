#include <pgr/neural/network.hpp>
#include <exception>
#include <iomanip>
#include <iostream>

namespace {
void print_vector(const char* name, const pgr::neural::Vector& values) {
    std::cout << name << ": [";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        } else {
            // Перед первым числом разделитель не нужен.
        }
        std::cout << values[i];
    }
    std::cout << "]\n";
}
} // namespace

int main() {
    try {
        const auto network = pgr::neural::make_example_network();
        const auto trace = network.forward({1, 2}, 1, 0);
        // ASCII-вывод не зависит от кодовой страницы консоли Windows.
        std::cout << std::setprecision(15);
        std::cout << "PGR Neural CPU forward 0.4.0 - fixed weights, NOT a trained model\n";
        print_vector("mean_embedding", trace.mean_embedding);
        print_vector("input", trace.input);
        print_vector("hidden_before_relu", trace.hidden_before_relu);
        print_vector("hidden", trace.hidden);
        print_vector("logits", trace.logits);
        print_vector("softmax", trace.probabilities);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
