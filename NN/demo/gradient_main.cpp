#include <pgr/neural/gradients.hpp>
#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const auto network = pgr::neural::make_example_network();
        const auto result = pgr::neural::backward(network, {{1,2},1,0,0});
        std::cout << std::setprecision(15);
        std::cout << "PGR Neural 0.4.0 - analytical gradients, NO training\n";
        std::cout << "loss: " << result.loss << '\n';
        std::cout << "d_b2: [" << result.gradients.b2[0] << ", " << result.gradients.b2[1] << "]\n";
        std::cout << "d_W2[0,0]: " << result.gradients.w2.at(0,0) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
