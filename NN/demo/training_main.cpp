#include <pgr/neural/trainer.hpp>
#include <pgr/neural/model_io.hpp>
#include "arguments.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace {
using namespace pgr::neural;
void json_vector(std::ostream& out, const Vector& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) { out << ','; }
        else { /* Первый элемент без запятой. */ }
        out << values[i];
    }
    out << ']';
}

void write_report(const std::filesystem::path& folder, const TrainResult& r,
                  const TrainOptions& o, std::uint64_t seed) {
    if (!std::filesystem::create_directory(folder)) {
        throw std::runtime_error("Output directory must be NEW");
    } else {
        // Не перезаписываем прежний опыт; родитель должен существовать.
    }
    save_model_new(r.network, folder / "model.pgrnn");
    const auto loaded = load_model(folder / "model.pgrnn");
    if (serialize_model(loaded) != serialize_model(r.network)) {
        throw std::runtime_error("Saved weights differ from trained weights");
    } else {
        // Проверяем не только массивы, но и вычисления всех учебных примеров.
    }
    for (const auto& sample : make_toy_dataset()) {
        const auto a = r.network.forward(sample.tokens, sample.compiler_index, sample.stage_index);
        const auto b = loaded.forward(sample.tokens, sample.compiler_index, sample.stage_index);
        if (a.logits != b.logits || a.probabilities != b.probabilities) {
            throw std::runtime_error("Saved model predictions differ");
        } else {
            // В одном бинарнике сравнение точное; tolerance здесь не нужен.
        }
    }
    std::ofstream csv(folder / "history.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << std::setprecision(std::numeric_limits<double>::max_digits10);
    csv << "epoch,updates,training_loss,training_accuracy\n";
    for (const auto& e : r.history) {
        csv << e.epoch << ',' << e.updates << ',' << e.training_loss << ',' << e.training_accuracy << '\n';
    }
    csv.close();
    const auto probe = loaded.forward({1, 2}, 0, 0);
    std::ofstream json(folder / "run.json");
    json.exceptions(std::ios::badbit | std::ios::failbit);
    json << std::setprecision(std::numeric_limits<double>::max_digits10);
    json << "{\n  \"version\": \"0.4.0\",\n  \"task\": \"toy_token_groups\",\n  \"training_samples\": 8,\n"
         << "  \"initialization_seed\": " << seed << ",\n  \"shuffle_seed\": " << o.shuffle_seed
         << ",\n  \"epochs\": " << o.epochs << ",\n  \"batch_size\": " << o.batch_size
         << ",\n  \"learning_rate\": " << o.learning_rate << ",\n  \"updates\": " << r.updates
         << ",\n  \"initial_training_loss\": " << r.history.front().training_loss
         << ",\n  \"final_training_loss\": " << r.history.back().training_loss
         << ",\n  \"final_training_accuracy\": " << r.history.back().training_accuracy
         << ",\n  \"independent_evaluation\": false,\n  \"weights_saved\": true,\n"
         << "  \"model_file\": \"model.pgrnn\",\n  \"model_format_version\": " << model_format_version
         << ",\n  \"model_roundtrip_exact\": true,\n  \"training_resume_supported\": false,\n"
         << "  \"probe\": {\"tokens\": [1,2], \"compiler_index\": 0, \"stage_index\": 0, \"logits\": ";
    json_vector(json, probe.logits);
    json << ", \"probabilities\": ";
    json_vector(json, probe.probabilities);
    json << "}\n}\n";
    json.close();
}
} // namespace

int main(int argc, char** argv) {
    using namespace pgr::neural;
    try {
        const auto args = cli::arguments(argc, argv);
        TrainOptions o;
        std::uint64_t seed = 20261028;
        std::filesystem::path output;
        std::set<std::string> seen;
        for (std::size_t i = 1; i < args.size(); ++i) {
            const auto& key = args[i];
            if (key == "--help") {
                std::cout << "Options: --seed N --epochs N --batch-size N --learning-rate R --output NEW_DIRECTORY\n";
                return 0;
            } else if (!seen.insert(key).second) {
                throw std::invalid_argument("Duplicate option: " + key);
            } else if (i + 1 >= args.size()) {
                throw std::invalid_argument("Missing option value");
            } else {
                const auto& value = args[++i];
                if (key == "--seed") {
                    seed = cli::integer(value);
                } else if (key == "--epochs") {
                    o.epochs = cli::size_value(value);
                } else if (key == "--batch-size") {
                    o.batch_size = cli::size_value(value);
                } else if (key == "--learning-rate") {
                    std::size_t consumed = 0;
                    o.learning_rate = std::stod(value, &consumed);
                    if (consumed != value.size()) {
                        throw std::invalid_argument("Invalid learning rate");
                    } else {
                        // Конечность и знак проверит train_sgd.
                    }
                } else if (key == "--output") {
                    if (value.empty()) {
                        throw std::invalid_argument("Output path is empty");
                    } else {
                        output = std::filesystem::u8path(value);
                    }
                } else {
                    throw std::invalid_argument("Unknown option: " + key);
                }
            }
        }
        if (!output.empty() && std::filesystem::exists(output)) {
            throw std::runtime_error("Output directory already exists");
        } else {
            // Обучение не запускается, если выбран занятый каталог.
        }
        o.shuffle_seed = seed;
        const auto trained = train_sgd(initialize_network(toy_shape(), seed), make_toy_dataset(), o);
        std::cout << std::setprecision(15);
        std::cout << "PGR Neural 0.4.0 - SGD TOY DEMO; NOT diagnostic corpus training\n"
                  << "seed: " << seed << "\ninitial_training_loss: " << trained.history.front().training_loss
                  << "\nfinal_training_loss: " << trained.history.back().training_loss
                  << "\nfinal_training_accuracy: " << trained.history.back().training_accuracy
                  << "\nupdates: " << trained.updates << '\n';
        if (!output.empty()) {
            write_report(output, trained, o, seed);
            std::cout << "model_saved: model.pgrnn\nmodel_roundtrip_exact: true\n";
        } else {
            // Без --output опыт выполняется только в памяти.
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
