#include <pgr/neural/trainer.hpp>
#include <cmath>
#include <stdexcept>

namespace pgr::neural {
namespace {
Real step(Real value, Real gradient, Real rate) {
    const Real change=rate*gradient;
    const Real updated=value-change;
    if (!std::isfinite(change) || !std::isfinite(updated)) {
        throw std::overflow_error("SGD update overflow");
    } else {
        return updated;
    }
}
void same(const Matrix& a, const Matrix& b) {
    if (a.rows()!=b.rows() || a.columns()!=b.columns()) {
        throw std::invalid_argument("Gradient matrix dimension mismatch");
    } else {
        // Matrix уже проверила конечность своих элементов в конструкторе.
    }
}
Matrix updated(const Matrix& p, const Matrix& g, Real rate) {
    Vector values;
    values.reserve(p.rows()*p.columns());
    for (std::size_t i=0;i<p.rows();++i) {
        for (std::size_t j=0;j<p.columns();++j) {
            values.push_back(step(p.at(i,j),g.at(i,j),rate));
        }
    }
    return Matrix(p.rows(),p.columns(),std::move(values));
}
Vector updated(const Vector& p, const Vector& g, Real rate) {
    Vector out(p.size());
    for (std::size_t i=0;i<p.size();++i) { out[i]=step(p[i],g[i],rate); }
    return out;
}
} // namespace

Network sgd_step(const Network& network, const Gradients& g, Real rate) {
    if (!std::isfinite(rate) || rate<=0.0) {
        throw std::invalid_argument("Learning rate must be finite and positive");
    } else {
        // Проверяем всё до вычисления новых параметров.
    }
    const auto& p=network.parameters();
    same(p.embeddings,g.embeddings);same(p.w1,g.w1);same(p.w2,g.w2);
    if (p.b1.size()!=g.b1.size() || p.b2.size()!=g.b2.size()) {
        throw std::invalid_argument("Gradient bias dimension mismatch");
    } else {
        require_finite(g.b1);require_finite(g.b2);
    }
    return Network(network.shape(),Parameters{updated(p.embeddings,g.embeddings,rate),updated(p.w1,g.w1,rate),
        updated(p.b1,g.b1,rate),updated(p.w2,g.w2,rate),updated(p.b2,g.b2,rate)});
}
} // namespace pgr::neural
