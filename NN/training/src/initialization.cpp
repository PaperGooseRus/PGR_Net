#include <pgr/neural/trainer.hpp>
#include "random.hpp"
#include <cmath>
#include <stdexcept>

namespace pgr::neural {
namespace {
void matrix_limit(std::size_t rows, std::size_t columns) {
    if (columns == 0 || rows > Matrix::max_elements / columns) {
        throw std::length_error("Initialized matrix exceeds element limit");
    } else {
        // Проверка сделана до выделения и умножения размеров.
    }
}
Matrix random_matrix(std::size_t rows, std::size_t columns, Real scale, detail::Random& rng) {
    Vector values(rows*columns,0.0);
    for (auto& value : values) {
        value = (2.0*rng.unit()-1.0)*scale;
    }
    return Matrix(rows,columns,std::move(values));
}
} // namespace

Network initialize_network(Shape s, std::uint64_t seed) {
    validate_shape(s);
    const auto input = s.embedding_dimension+s.compiler_count+s.stage_count;
    matrix_limit(s.vocabulary,s.embedding_dimension);
    matrix_limit(s.hidden_size,input);
    matrix_limit(s.class_count,s.hidden_size);
    detail::Random rng(seed);
    auto embeddings=random_matrix(s.vocabulary,s.embedding_dimension,0.5,rng);
    auto w1=random_matrix(s.hidden_size,input,std::sqrt(6.0/static_cast<Real>(input+s.hidden_size)),rng);
    auto w2=random_matrix(s.class_count,s.hidden_size,std::sqrt(6.0/static_cast<Real>(s.hidden_size+s.class_count)),rng);
    return Network(s,Parameters{std::move(embeddings),std::move(w1),Vector(s.hidden_size,0.1),
                                std::move(w2),Vector(s.class_count,0.0)});
}
} // namespace pgr::neural
