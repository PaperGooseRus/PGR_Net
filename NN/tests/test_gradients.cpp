#include <pgr/neural/gradients.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace pgr::neural;

void check(bool ok,const std::string& text) {
    if (!ok) { throw std::runtime_error(text); } else { /* Проверка пройдена. */ }
}
void near(Real actual, Real expected, Real atol=1e-10, Real rtol=1e-8) {
    check(std::isfinite(actual) && std::abs(actual-expected)<=atol+rtol*std::abs(expected),
          "Mismatch actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
template<class Exception,class Action> void throws(Action action) {
    bool caught=false;
    try { action(); } catch (const Exception&) { caught=true; }
    check(caught,"Expected exception missing");
}
void append(Vector& dest,const Matrix& m) {
    for (std::size_t i=0;i<m.rows();++i) {
        for (std::size_t j=0;j<m.columns();++j) { dest.push_back(m.at(i,j)); }
    }
}
Vector flatten(const Parameters& p) {
    Vector v; append(v,p.embeddings); append(v,p.w1);
    v.insert(v.end(),p.b1.begin(),p.b1.end()); append(v,p.w2);
    v.insert(v.end(),p.b2.begin(),p.b2.end()); return v;
}
Vector flatten(const Gradients& g) {
    Vector v; append(v,g.embeddings); append(v,g.w1);
    v.insert(v.end(),g.b1.begin(),g.b1.end()); append(v,g.w2);
    v.insert(v.end(),g.b2.begin(),g.b2.end()); return v;
}
Parameters parameters(const Shape& s,const Vector& values) {
    std::size_t pos=0;
    auto take=[&](std::size_t count) {
        check(pos+count<=values.size(),"Invalid flattened parameter size");
        Vector out(values.begin()+pos,values.begin()+pos+count);pos+=count;return out;
    };
    const auto input=s.embedding_dimension+s.compiler_count+s.stage_count;
    auto e=take(s.vocabulary*s.embedding_dimension);
    auto w1=take(s.hidden_size*input); auto b1=take(s.hidden_size);
    auto w2=take(s.class_count*s.hidden_size); auto b2=take(s.class_count);
    check(pos==values.size(),"Unused flattened data");
    return {Matrix(s.vocabulary,s.embedding_dimension,std::move(e)),Matrix(s.hidden_size,input,std::move(w1)),
            std::move(b1),Matrix(s.class_count,s.hidden_size,std::move(w2)),std::move(b2)};
}

void check_numerical(const Network& network,const std::vector<LabeledInput>& samples) {
    const auto base=flatten(network.parameters());
    const auto analytic=flatten(backward_batch(network,samples).gradients);
    constexpr Real epsilon=1e-6;
    Real worst_absolute=0;
    for (std::size_t j=0;j<base.size();++j) {
        auto plus=base; auto minus=base;
        plus[j]+=epsilon;minus[j]-=epsilon;
        const Network nplus(network.shape(),parameters(network.shape(),plus));
        const Network nminus(network.shape(),parameters(network.shape(),minus));
        // Не сравниваем центральную разность через излом ReLU.
        Real positive=0,negative=0;
        for (const auto& sample:samples) {
            const auto a=nplus.forward(sample.tokens,sample.compiler_index,sample.stage_index);
            const auto b=nminus.forward(sample.tokens,sample.compiler_index,sample.stage_index);
            for (std::size_t i=0;i<a.hidden_before_relu.size();++i) {
                check(a.hidden_before_relu[i]*b.hidden_before_relu[i]>0,"Gradient test crossed ReLU kink");
            }
            positive+=loss(nplus,sample);negative+=loss(nminus,sample);
        }
        const auto numerical=(positive-negative)/(2*epsilon*static_cast<Real>(samples.size()));
        worst_absolute=std::max(worst_absolute,std::abs(analytic[j]-numerical));
        near(analytic[j],numerical,2e-7,2e-5);
    }
    std::cout<<"GRADCHECK parameters="<<base.size()<<" max_absolute_error="<<worst_absolute<<'\n';
}

void manual_loss_and_gradient() {
    const auto n=make_example_network();
    const auto r=backward(n,{{1,2},1,0,0});
    near(r.loss,0.1269280110429725);
    near(r.gradients.b2[0],-0.1192029220221176);
    near(r.gradients.b2[1], 0.1192029220221176);
    near(r.gradients.w2.at(0,0),-0.2384058440442352);
    near(r.gradients.w2.at(1,0), 0.2384058440442352);
    near(r.gradients.b1[0],-0.1192029220221176);
    near(r.gradients.b1[1],0);
    for (std::size_t d=0;d<2;++d) {
        near(r.gradients.embeddings.at(0,d),0);
        near(r.gradients.embeddings.at(1,d),-0.0596014610110588);
        near(r.gradients.embeddings.at(2,d),-0.0596014610110588);
    }
}
void stable_loss_underflow() {
    near(cross_entropy({1000,-1000},1),2000);
    near(cross_entropy({1000,-1000},0),0);
    near(cross_entropy({-1000,-1000},0),0.6931471805599453);
    near(cross_entropy({123},0),0);
}
void loss_shift_invariant() {
    near(cross_entropy({2,0},1),cross_entropy({1002,1000},1));
}
void numerical_single() { check_numerical(make_example_network(),{{{1,2},1,0,0}}); }
void numerical_other_target_context() { check_numerical(make_example_network(),{{{1,2},0,1,1}}); }
void numerical_repeated_tokens() { check_numerical(make_example_network(),{{{1,2,1},1,0,1}}); }
void numerical_both_active() {
    const auto example=make_example_network();
    auto p=example.parameters();p.b1[1]=2;
    check_numerical(Network(example.shape(),std::move(p)),{{{1,2},1,0,0}});
}
void numerical_batch() {
    check_numerical(make_example_network(),{{{1,2},1,0,0},{{1,2,1},0,1,1}});
}
void batch_is_mean() {
    const auto n=make_example_network();
    const auto a=backward(n,{{1,2},1,0,0});
    const auto b=backward(n,{{1,2},1,0,1});
    const auto mean=backward_batch(n,{{{1,2},1,0,0},{{1,2},1,0,1}});
    near(mean.loss,1.1269280110429725);
    const auto av=flatten(a.gradients),bv=flatten(b.gradients),mv=flatten(mean.gradients);
    for (std::size_t i=0;i<mv.size();++i) { near(mv[i],(av[i]+bv[i])/2); }
}
void relu_zero_convention() {
    const auto base=make_example_network();auto p=base.parameters();
    p.w1=Matrix(2,6,Vector(12,0));p.b1={0,0};
    const Network n(base.shape(),std::move(p));
    const auto r=backward(n,{{1,2},1,0,0});
    near(r.loss,0.6931471805599453);
    for (std::size_t i=0;i<2;++i) {
        near(r.gradients.b1[i],0);
        for (std::size_t j=0;j<6;++j) { near(r.gradients.w1.at(i,j),0); }
    }
    for (std::size_t i=0;i<3;++i) {
        for (std::size_t j=0;j<2;++j) { near(r.gradients.embeddings.at(i,j),0); }
    }
}
void weights_not_modified() {
    const auto n=make_example_network();const auto before=flatten(n.parameters());
    static_cast<void>(backward(n,{{1,2},1,0,0}));
    check(before==flatten(n.parameters()),"backward modified parameters");
}
void arbitrary_class_counts() {
    for (const std::size_t k : {std::size_t{1},std::size_t{7},std::size_t{117}}) {
        const Network n(Shape{1,1,1,1,1,k,1},Parameters{
            Matrix(1,1,{0}),Matrix(1,3,{0,0,0}),{0},Matrix(k,1,Vector(k,0)),Vector(k,0)});
        const auto r=backward(n,{{0},0,0,k-1});
        near(r.loss,std::log(static_cast<Real>(k)));
        for (std::size_t i=0;i<k;++i) {
            const Real expected=1.0/static_cast<Real>(k)-(i==k-1?1.0:0.0);
            near(r.gradients.b2[i],expected);
        }
    }
}
void rejected_inputs() {
    const auto n=make_example_network();
    throws<std::invalid_argument>([]{static_cast<void>(cross_entropy({},0));});
    throws<std::invalid_argument>([]{static_cast<void>(cross_entropy({1,2},2));});
    throws<std::invalid_argument>([&]{static_cast<void>(backward(n,{{1,2},1,0,2}));});
    throws<std::invalid_argument>([&]{static_cast<void>(backward_batch(n,{}));});
    throws<std::invalid_argument>([&]{static_cast<void>(backward_batch(n,std::vector<LabeledInput>(257,{{1},0,0,0})));});
    throws<std::invalid_argument>([]{static_cast<void>(cross_entropy({std::numeric_limits<Real>::infinity()},0));});
    throws<std::invalid_argument>([]{static_cast<void>(cross_entropy({std::numeric_limits<Real>::quiet_NaN()},0));});
}
} // namespace

int main() {
    const std::vector<std::pair<const char*,std::function<void()>>> tests={
        {"manual_loss_and_gradient",manual_loss_and_gradient},{"stable_loss_underflow",stable_loss_underflow},
        {"loss_shift_invariant",loss_shift_invariant},{"numerical_single_all_parameters",numerical_single},
        {"numerical_target_context",numerical_other_target_context},{"numerical_repeated_tokens",numerical_repeated_tokens},
        {"numerical_both_relu_active",numerical_both_active},{"numerical_batch",numerical_batch},
        {"batch_is_mean",batch_is_mean},{"relu_zero_convention",relu_zero_convention},
        {"weights_not_modified",weights_not_modified},{"arbitrary_class_counts",arbitrary_class_counts},{"rejected_inputs",rejected_inputs}};
    std::size_t failures=0;
    for (const auto& test:tests) {
        try {test.second();std::cout<<"PASS "<<test.first<<'\n';}
        catch(const std::exception& error) {++failures;std::cerr<<"FAIL "<<test.first<<": "<<error.what()<<'\n';}
    }
    std::cout<<"Gradient tests: "<<tests.size()<<"; failures: "<<failures<<'\n';
    return failures==0?0:1;
}
