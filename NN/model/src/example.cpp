#include <pgr/neural/network.hpp>

namespace pgr::neural {

Network make_example_network() {
    // Индекс токена 0 резервируем под <unk>. Имена классов не присваиваются:
    // демонстрационные числа не являются обученной диагностикой ошибок.
    return Network(
        Shape{3, 2, 2, 2, 2, 2, 128},
        Parameters{
            Matrix(3, 2, {0.0, 0.0,  2.0, 0.0,  0.0, 2.0}),
            Matrix(2, 6, { 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                         -1.0, 0.0, 0.0, 1.0, 0.0, 0.0}),
            Vector{-1.0, -1.0},
            Matrix(2, 2, {1.0, 0.0, 0.0, 1.0}),
            Vector{0.0, 0.0}
        });
}

} // namespace pgr::neural
