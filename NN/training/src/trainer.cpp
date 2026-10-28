#include <pgr/neural/trainer.hpp>
#include "random.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace pgr::neural {
namespace {
EpochMetrics measure(const Network& network,const std::vector<LabeledInput>& data,
                     std::size_t epoch,std::size_t updates) {
    Real total=0.0;
    std::size_t correct=0;
    for (const auto& sample:data) {
        const auto trace=network.forward(sample.tokens,sample.compiler_index,sample.stage_index);
        total+=cross_entropy(trace.logits,sample.target_class);
        if (!std::isfinite(total)) {
            throw std::overflow_error("Training loss accumulation overflow");
        } else {
            // Из равных оценок max_element выбирает первый, меньший индекс.
        }
        const auto predicted=static_cast<std::size_t>(std::max_element(trace.logits.begin(),trace.logits.end())-trace.logits.begin());
        if (predicted==sample.target_class) { ++correct; } else { /* Ошибка на учебном наборе. */ }
    }
    return {epoch,updates,total/static_cast<Real>(data.size()),static_cast<Real>(correct)/static_cast<Real>(data.size())};
}
} // namespace

TrainResult train_sgd(Network network,const std::vector<LabeledInput>& data,const TrainOptions& o) {
    if (data.empty() || data.size()>100000 || o.batch_size==0 || o.batch_size>256 || o.epochs==0 || o.epochs>10000 ||
        !std::isfinite(o.learning_rate) || o.learning_rate<=0.0) {
        throw std::invalid_argument("Invalid training options or dataset size");
    } else {
        // Лимиты исключают непреднамеренно огромный учебный запуск.
    }
    const std::size_t batches=(data.size()+o.batch_size-1)/o.batch_size;
    if (o.epochs>100000/batches) {
        throw std::length_error("Training exceeds 100000 update limit");
    } else {
        // Все данные проверяем до первого шага, не только первый батч.
    }
    std::vector<EpochMetrics> history;
    history.push_back(measure(network,data,0,0));
    std::vector<std::size_t> order(data.size());
    detail::Random rng(o.shuffle_seed);
    std::size_t updates=0;
    for (std::size_t epoch=1;epoch<=o.epochs;++epoch) {
        std::iota(order.begin(),order.end(),0);
        if (o.shuffle) {
            for (std::size_t i=order.size();i>1;--i) { std::swap(order[i-1],order[rng.bounded(i)]); }
        } else {
            // Сохраняем исходный порядок для ручных проверок и опытов.
        }
        for (std::size_t start=0;start<order.size();start+=o.batch_size) {
            const auto end=std::min(start+o.batch_size,order.size());
            std::vector<LabeledInput> batch;
            batch.reserve(end-start);
            for (auto i=start;i<end;++i) { batch.push_back(data[order[i]]); }
            const auto result=backward_batch(network,batch);
            network=sgd_step(network,result.gradients,o.learning_rate);
            ++updates;
        }
        history.push_back(measure(network,data,epoch,updates));
    }
    return {std::move(network),std::move(history),updates};
}

Shape toy_shape() { return {5,4,2,2,8,2,16}; }
std::vector<LabeledInput> make_toy_dataset() {
    return {{{1},0,0,0},{{2},0,0,0},{{1,2},0,0,0},{{2,1,1},0,0,0},
            {{3},0,0,1},{{4},0,0,1},{{3,4},0,0,1},{{4,3,3},0,0,1}};
}
} // namespace pgr::neural
