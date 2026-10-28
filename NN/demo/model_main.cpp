#include <pgr/neural/model_io.hpp>
#include "arguments.hpp"
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace {
using namespace pgr::neural;
std::vector<std::size_t> tokens(const std::string& text) {
    std::vector<std::size_t> result;
    std::size_t begin = 0;
    while (true) {
        const auto end = text.find(',', begin);
        result.push_back(cli::size_value(text.substr(begin, end == std::string::npos ? end : end - begin)));
        if (result.size() > 4096) {
            throw std::length_error("Too many token indices");
        } else if (end == std::string::npos) {
            return result;
        } else {
            begin = end + 1;
        }
    }
}
template<class T> void json_array(const std::vector<T>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) { std::cout << ','; }
        else { /* Первый элемент. */ }
        std::cout << values[i];
    }
    std::cout << ']';
}
} // namespace

int main(int argc, char** argv) {
    using namespace pgr::neural;
    try {
        const auto args = cli::arguments(argc, argv);
        std::set<std::string> seen;
        std::filesystem::path input, copy;
        std::vector<std::size_t> ids;
        std::size_t compiler = 0, stage = 0;
        for (std::size_t i = 1; i < args.size(); ++i) {
            const auto& key = args[i];
            if (key == "--help") {
                std::cout << "Required: --input FILE --tokens 1,2 --compiler INDEX --stage INDEX\n"
                             "Optional: --copy-to NEW_FILE (validated model only; never overwrite)\n";
                return 0;
            } else if (!seen.insert(key).second) {
                throw std::invalid_argument("Duplicate option: " + key);
            } else if (i + 1 >= args.size()) {
                throw std::invalid_argument("Missing option value");
            } else {
                const auto& value = args[++i];
                if (key == "--input" || key == "--copy-to") {
                    if (value.empty()) {
                        throw std::invalid_argument("Model path is empty");
                    } else if (key == "--input") {
                        input = std::filesystem::u8path(value);
                    } else {
                        copy = std::filesystem::u8path(value);
                    }
                } else if (key == "--tokens") {
                    ids = tokens(value);
                } else if (key == "--compiler") {
                    compiler = cli::size_value(value);
                } else if (key == "--stage") {
                    stage = cli::size_value(value);
                } else {
                    throw std::invalid_argument("Unknown option: " + key);
                }
            }
        }
        for (const auto* key : {"--input", "--tokens", "--compiler", "--stage"}) {
            if (seen.count(key) == 0) {
                throw std::invalid_argument(std::string("Required option missing: ") + key);
            } else {
                // Нет неявного подбора контекста или классов для чужой модели.
            }
        }
        const auto network = load_model(input);
        const auto trace = network.forward(ids, compiler, stage);
        if (!copy.empty()) {
            save_model_new(network, copy);
        } else {
            // Обычный режим inference ничего не записывает на диск.
        }
        const auto& s = network.shape();
        std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10);
        std::cout << "{\n  \"version\": \"0.4.0\",\n  \"format_version\": " << model_format_version
                  << ",\n  \"shape\": [" << s.vocabulary << ',' << s.embedding_dimension << ','
                  << s.compiler_count << ',' << s.stage_count << ',' << s.hidden_size << ','
                  << s.class_count << ',' << s.max_tokens << "],\n  \"tokens\": ";
        json_array(ids);
        std::cout << ",\n  \"compiler_index\": " << compiler << ",\n  \"stage_index\": " << stage
                  << ",\n  \"logits\": ";
        json_array(trace.logits);
        std::cout << ",\n  \"probabilities\": ";
        json_array(trace.probabilities);
        std::cout << "\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
