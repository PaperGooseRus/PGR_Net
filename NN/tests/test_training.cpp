#include <pgr/neural/trainer.hpp>
#include "../training/src/random.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace pgr::neural;
void check(bool value,const char* text) {
    if (!value) {throw std::runtime_error(text);} else { /* Проверка выполнена. */ }
}
void near(Real value,Real expected,Real tolerance=1e-12) {
    check(std::isfinite(value)&&std::abs(value-expected)<=tolerance,"Numeric mismatch");
}
template<class E,class Action> void throws(Action action) {
    bool caught=false;try {action();}catch(const E&){caught=true;}check(caught,"Exception missing");
}
void append(Vector& v,const Matrix& m) {
    for (std::size_t i=0;i<m.rows();++i) {
        for (std::size_t j=0;j<m.columns();++j) {v.push_back(m.at(i,j));}
    }
}
Vector flatten(const Network& n) {
    const auto& p=n.parameters();Vector v;append(v,p.embeddings);append(v,p.w1);
    v.insert(v.end(),p.b1.begin(),p.b1.end());append(v,p.w2);v.insert(v.end(),p.b2.begin(),p.b2.end());return v;
}
void prng_known_bits() {
    detail::Random rng(0);
    check(rng.next()==UINT64_C(0xe220a8397b1dcdaf),"PRNG reference bits mismatch");
}
void seeded_initialization() {
    const auto a=initialize_network(toy_shape(),1234),b=initialize_network(toy_shape(),1234);
    const auto c=initialize_network(toy_shape(),1235);
    check(flatten(a)==flatten(b),"Same seed initialization differs");
    check(flatten(a)!=flatten(c),"Chosen different seeds unexpectedly match");
}
void manual_sgd() {
    const auto n=make_example_network();const auto old=flatten(n);
    const auto g=backward(n,{{1,2},1,0,0}).gradients;
    const auto after=sgd_step(n,g,0.1);const auto& p=after.parameters();
    near(p.embeddings.at(1,0),2.0059601461011059);
    near(p.w1.at(0,0),1.0119202922022118);
    near(p.b1[0],-0.9880797077977882);
    near(p.w2.at(0,0),1.0238405844044235);
    near(p.b2[0],0.0119202922022118);
    check(flatten(n)==old,"SGD modified original network");
    check(loss(after,{{1,2},1,0,0})<loss(n,{{1,2},1,0,0}),"Small SGD step did not lower sample loss");
}
void rejected_updates_are_transactional() {
    const auto n=make_example_network();const auto old=flatten(n);auto g=backward(n,{{1,2},1,0,0}).gradients;
    for (Real rate:{0.0,-1.0,std::numeric_limits<Real>::infinity(),std::numeric_limits<Real>::quiet_NaN()}) {
        throws<std::invalid_argument>([&]{static_cast<void>(sgd_step(n,g,rate));});
    }
    g.b1[0]=std::numeric_limits<Real>::quiet_NaN();
    throws<std::invalid_argument>([&]{static_cast<void>(sgd_step(n,g,0.1));});
    g=backward(n,{{1,2},1,0,0}).gradients;g.w1=Matrix(1,1,{0});
    throws<std::invalid_argument>([&]{static_cast<void>(sgd_step(n,g,0.1));});
    g=backward(n,{{1,2},1,0,0}).gradients;g.b2[0]=std::numeric_limits<Real>::max();
    throws<std::overflow_error>([&]{static_cast<void>(sgd_step(n,g,2));});
    check(flatten(n)==old,"Rejected SGD changed original parameters");
}
void seeded_training_repeatable() {
    TrainOptions o;o.epochs=15;
    const auto initial=initialize_network(toy_shape(),20261028);const auto old=flatten(initial);
    const auto a=train_sgd(initial,make_toy_dataset(),o),b=train_sgd(initial,make_toy_dataset(),o);
    check(flatten(a.network)==flatten(b.network),"Repeated training differs on same build");
    check(a.history.size()==b.history.size(),"History size mismatch");
    for (std::size_t i=0;i<a.history.size();++i) {
        check(a.history[i].training_loss==b.history[i].training_loss,"Repeated loss differs");
    }
    check(flatten(initial)==old,"Training changed caller network");
}
void toy_converges() {
    const auto r=train_sgd(initialize_network(toy_shape(),20261028),make_toy_dataset(),TrainOptions{});
    check(r.history.size()==81&&r.updates==160,"Wrong epoch/update count");
    check(r.history.back().training_loss<r.history.front().training_loss*0.2,"Toy loss reduction insufficient");
    near(r.history.back().training_accuracy,1.0);
    for (const auto& e:r.history) { check(std::isfinite(e.training_loss)&&e.training_loss>=0,"Invalid epoch loss"); }
    std::cout<<"TOY initial_loss="<<r.history.front().training_loss<<" final_loss="<<r.history.back().training_loss<<'\n';
}
void last_batch_uses_actual_size() {
    const auto n=initialize_network(toy_shape(),12);
    const auto all=make_toy_dataset();const std::vector<LabeledInput> data(all.begin(),all.begin()+5);
    TrainOptions o;o.epochs=1;o.batch_size=3;o.shuffle=false;
    const auto result=train_sgd(n,data,o);
    const std::vector<LabeledInput> first(data.begin(),data.begin()+3),last(data.begin()+3,data.end());
    const auto one=sgd_step(n,backward_batch(n,first).gradients,o.learning_rate);
    const auto two=sgd_step(one,backward_batch(one,last).gradients,o.learning_rate);
    check(flatten(result.network)==flatten(two),"Partial batch incorrectly weighted or dropped");
    check(result.updates==2,"Partial batch not processed");
}
void batch_larger_than_dataset() {
    auto data=make_toy_dataset();data.resize(2);
    const auto n=initialize_network(toy_shape(),99);
    TrainOptions o;o.epochs=1;o.batch_size=8;o.shuffle=false;
    const auto trained=train_sgd(n,data,o);
    const auto expected=sgd_step(n,backward_batch(n,data).gradients,o.learning_rate);
    check(flatten(trained.network)==flatten(expected)&&trained.updates==1,"Large batch mismatch");
}
void invalid_training_options() {
    const auto n=initialize_network(toy_shape(),3);auto data=make_toy_dataset();TrainOptions o;
    throws<std::invalid_argument>([&]{static_cast<void>(train_sgd(n,{},o));});
    o.epochs=0;throws<std::invalid_argument>([&]{static_cast<void>(train_sgd(n,data,o));});
    o=TrainOptions{};o.batch_size=0;throws<std::invalid_argument>([&]{static_cast<void>(train_sgd(n,data,o));});
    o=TrainOptions{};o.learning_rate=-1;throws<std::invalid_argument>([&]{static_cast<void>(train_sgd(n,data,o));});
    o=TrainOptions{};o.epochs=10000;o.batch_size=1;
    data.resize(11,data[0]);throws<std::length_error>([&]{static_cast<void>(train_sgd(n,data,o));});
    data=make_toy_dataset();data.back().target_class=99;o=TrainOptions{};
    throws<std::invalid_argument>([&]{static_cast<void>(train_sgd(n,data,o));});
}
void invalid_initialization_shape() {
    auto s=toy_shape();s.class_count=0;
    throws<std::invalid_argument>([&]{static_cast<void>(initialize_network(s,1));});
    s=toy_shape();s.vocabulary=Matrix::max_elements;
    throws<std::length_error>([&]{static_cast<void>(initialize_network(s,1));});
}
} // namespace

int main() {
    const std::vector<std::pair<const char*,std::function<void()>>> tests={
        {"prng_known_bits",prng_known_bits},{"seeded_initialization",seeded_initialization},
        {"manual_sgd",manual_sgd},{"rejected_updates_transactional",rejected_updates_are_transactional},
        {"seeded_training_repeatable",seeded_training_repeatable},{"toy_converges",toy_converges},
        {"last_batch_actual_size",last_batch_uses_actual_size},{"batch_larger_than_dataset",batch_larger_than_dataset},
        {"invalid_training_options",invalid_training_options},{"invalid_initialization_shape",invalid_initialization_shape}};
    std::size_t failures=0;
    for (const auto& test:tests) {
        try {test.second();std::cout<<"PASS "<<test.first<<'\n';}
        catch(const std::exception& error) {++failures;std::cerr<<"FAIL "<<test.first<<": "<<error.what()<<'\n';}
    }
    std::cout<<"Training tests: "<<tests.size()<<"; failures: "<<failures<<'\n';
    return failures==0?0:1;
}
