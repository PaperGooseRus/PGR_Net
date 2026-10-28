#include <pgr/neural/model_io.hpp>
#include <pgr/neural/trainer.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace pgr::neural;
void check(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
    else { /* Условие выполнено. */ }
}
template<class Action> void rejects(Action action) {
    bool caught = false;
    try { action(); }
    catch (const std::invalid_argument&) { caught = true; }
    catch (const std::length_error&) { caught = true; }
    check(caught, "Malformed model was accepted or wrong exception type");
}
template<class Action> void io_rejects(Action action) {
    bool caught = false;
    try { action(); }
    catch (const std::invalid_argument&) { caught = true; }
    catch (const std::length_error&) { caught = true; }
    catch (const std::system_error&) { caught = true; }
    catch (const std::runtime_error&) { caught = true; }
    check(caught, "Invalid file operation succeeded");
}
void put(ModelBytes& bytes, std::size_t offset, std::uint64_t value, std::size_t count = 8) {
    for (std::size_t i = 0; i < count; ++i) {
        bytes.at(offset + i) = static_cast<std::uint8_t>(value & 255U);
        value >>= 8U;
    }
}
void fix_crc(ModelBytes& bytes) {
    // Независимая табличная реализация в тесте нужна, чтобы проверки NaN,
    // размеров и версии проходили сквозь корректный CRC, а не только ловили
    // случайную порчу. Двоичный golden-файл дополнительно создан Python zlib.
    std::array<std::uint32_t, 256> table{};
    for (std::size_t i = 0; i < table.size(); ++i) {
        auto value = static_cast<std::uint32_t>(i);
        for (unsigned bit = 0; bit < 8; ++bit) {
            value = (value >> 1U) ^ ((value & 1U) * UINT32_C(0xEDB88320));
        }
        table[i] = value;
    }
    std::uint32_t crc = UINT32_C(0xFFFFFFFF);
    for (std::size_t i = 0; i < bytes.size() - 4; ++i) {
        crc = table[(crc ^ bytes[i]) & 255U] ^ (crc >> 8U);
    }
    put(bytes, bytes.size() - 4, crc ^ UINT32_C(0xFFFFFFFF), 4);
}
ModelBytes read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    check(static_cast<bool>(input), "Cannot open test fixture");
    return ModelBytes(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}
void write_bytes(const std::filesystem::path& path, const ModelBytes& bytes) {
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();
}
class Folder final {
public:
    Folder() {
        for (unsigned i = 0; i < 64; ++i) {
            path = std::filesystem::temp_directory_path() / ("pgrnn-io-test-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                "-" + std::to_string(i));
            if (std::filesystem::create_directory(path)) { return; }
            else { /* Уже занято: пробуем другое имя. */ }
        }
        throw std::runtime_error("Cannot create storage test directory");
    }
    ~Folder() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    std::filesystem::path path;
};

void external_binary_reference() {
    const auto bytes = read_bytes("tests/fixtures/example_v1.pgrnn");
    check(bytes.size() == 340, "Reference file size changed");
    check(serialize_model(make_example_network()) == bytes, "Python golden reference mismatch");
    const auto result = deserialize_model(bytes).forward({1, 2}, 1, 0);
    check(result.logits == Vector({2, 0}), "Golden reference prediction mismatch");
}
void deterministic_bytes() {
    const auto a = initialize_network(toy_shape(), 20261028);
    check(serialize_model(a) == serialize_model(a), "Serialization is not deterministic");
    check(serialize_model(a) != serialize_model(initialize_network(toy_shape(), 20261029)),
          "Different parameter sets serialized identically");
}
void exact_all_traces() {
    const auto source = make_example_network();
    const auto bytes = serialize_model(source);
    const auto loaded = deserialize_model(bytes);
    for (std::size_t c = 0; c < 2; ++c) {
        for (std::size_t s = 0; s < 2; ++s) {
            const auto a = source.forward({1, 2, 1}, c, s), b = loaded.forward({1, 2, 1}, c, s);
            check(a.mean_embedding == b.mean_embedding && a.input == b.input &&
                  a.hidden_before_relu == b.hidden_before_relu && a.hidden == b.hidden &&
                  a.logits == b.logits && a.probabilities == b.probabilities, "Round-trip trace differs");
        }
    }
    check(serialize_model(loaded) == bytes, "Round-trip parameter bits differ");
}
void trained_weights_roundtrip() {
    const auto trained = train_sgd(initialize_network(toy_shape(), 20261028),
                                   make_toy_dataset(), TrainOptions{});
    const auto loaded = deserialize_model(serialize_model(trained.network));
    for (const auto& sample : make_toy_dataset()) {
        const auto a = trained.network.forward(sample.tokens, sample.compiler_index, sample.stage_index);
        const auto b = loaded.forward(sample.tokens, sample.compiler_index, sample.stage_index);
        check(a.logits == b.logits && a.probabilities == b.probabilities, "Trained prediction changed");
    }
}
void floating_point_bits_preserved() {
    const auto p = make_example_network().parameters();
    auto q = p;
    q.embeddings = Matrix(3, 2, {0.0, -0.0, std::numeric_limits<Real>::denorm_min(),
        -std::numeric_limits<Real>::denorm_min(), std::numeric_limits<Real>::max(),
        std::numeric_limits<Real>::lowest()});
    q.b1 = {-0.0, std::numeric_limits<Real>::min()};
    const Network source(make_example_network().shape(), q);
    const auto bytes = serialize_model(source);
    const auto loaded = deserialize_model(bytes);
    check(std::signbit(loaded.parameters().embeddings.at(0, 1)), "Negative zero lost");
    check(serialize_model(loaded) == bytes, "Finite floating-point bits changed");
}
void arbitrary_class_and_context_sizes() {
    for (const std::size_t k : {1U, 7U, 117U}) {
        auto shape = toy_shape();
        shape.class_count = k;
        const auto source = initialize_network(shape, k);
        const auto loaded = deserialize_model(serialize_model(source));
        const auto a = source.forward({1, 2}, 1, 1), b = loaded.forward({1, 2}, 1, 1);
        check(a.probabilities == b.probabilities && b.probabilities.size() == k, "Class count changed");
    }
}
void every_truncation_rejected() {
    const auto full = serialize_model(make_example_network());
    for (std::size_t count = 0; count < full.size(); ++count) {
        const ModelBytes shorter(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(count));
        rejects([&] { static_cast<void>(deserialize_model(shorter)); });
    }
}
void extra_bytes_rejected() {
    auto bytes = serialize_model(make_example_network());
    bytes.push_back(0);
    rejects([&] { static_cast<void>(deserialize_model(bytes)); });
}
void field_rejected(std::size_t offset, std::uint64_t value, std::size_t count = 4) {
    auto bytes = serialize_model(make_example_network());
    put(bytes, offset, value, count);
    fix_crc(bytes);
    rejects([&] { static_cast<void>(deserialize_model(bytes)); });
}
void magic_rejected() { field_rejected(0, 0, 1); }
void version_rejected() { field_rejected(8, 0); field_rejected(8, 2); }
void header_length_rejected() { field_rejected(12, 120); }
void scalar_type_rejected() { field_rejected(16, 2); }
void flags_rejected() { field_rejected(20, 1); }
void architecture_rejected() { field_rejected(104, 2); }
void reserved_rejected() { field_rejected(127, 1, 1); }
void zero_and_large_dimensions() {
    for (std::size_t i = 0; i < 7; ++i) { field_rejected(32 + i * 8, 0, 8); }
    for (std::size_t i = 1; i < 7; ++i) { field_rejected(32 + i * 8, 4097, 8); }
    field_rejected(32, UINT64_MAX, 8);
}
void matrix_product_limited_before_allocation() {
    auto bytes = serialize_model(make_example_network());
    put(bytes, 32, Matrix::max_elements);
    put(bytes, 40, 4096);
    fix_crc(bytes);
    rejects([&] { static_cast<void>(deserialize_model(bytes)); });
}
void declared_sizes_rejected() {
    for (const auto offset : {24U, 88U, 96U}) {
        field_rejected(offset, UINT64_MAX, 8);
        field_rejected(offset, 0, 8);
    }
}
void crc_covers_payload_and_header() {
    auto bytes = serialize_model(make_example_network());
    bytes[128] ^= 1U;
    rejects([&] { static_cast<void>(deserialize_model(bytes)); });
    bytes = serialize_model(make_example_network());
    // max_tokens=128 -> 129: валидный Shape с тем же размером файла.
    bytes[80] ^= 1U;
    rejects([&] { static_cast<void>(deserialize_model(bytes)); });
    bytes = serialize_model(make_example_network());
    bytes.back() ^= 1U;
    rejects([&] { static_cast<void>(deserialize_model(bytes)); });
}
void nan_inf_rejected_even_with_valid_crc() {
    // Первые элементы каждого из пяти массивов ручного примера.
    for (const auto index : {0U, 6U, 18U, 20U, 24U}) {
        for (const auto bits : {UINT64_C(0x7FF8000000000001), UINT64_C(0x7FF0000000000000),
                                UINT64_C(0xFFF0000000000000)}) {
            auto bytes = serialize_model(make_example_network());
            put(bytes, 128 + index * 8, bits);
            fix_crc(bytes);
            rejects([&] { static_cast<void>(deserialize_model(bytes)); });
        }
    }
}
void malformed_load_does_not_change_source() {
    const auto source = initialize_network(toy_shape(), 28);
    const auto before = serialize_model(source);
    auto bad = before;
    bad.pop_back();
    rejects([&] { static_cast<void>(deserialize_model(bad)); });
    check(serialize_model(source) == before, "Rejected load mutated existing model");
    check(bad.size() == before.size() - 1, "Deserializer modified input bytes");
}
void file_roundtrip_and_no_temp_left() {
    Folder folder;
    const auto model = make_example_network();
    save_model_new(model, folder.path / "model.pgrnn");
    check(serialize_model(load_model(folder.path / "model.pgrnn")) == serialize_model(model),
          "Disk round-trip differs");
    check(std::distance(std::filesystem::directory_iterator(folder.path),
                        std::filesystem::directory_iterator{}) == 1, "Temporary file leaked");
}
void existing_file_preserved() {
    Folder folder;
    const auto target = folder.path / "occupied.pgrnn";
    const ModelBytes sentinel = {1, 2, 3, 4, 255};
    write_bytes(target, sentinel);
    io_rejects([&] { save_model_new(make_example_network(), target); });
    check(read_bytes(target) == sentinel, "Existing file was overwritten");
    check(std::distance(std::filesystem::directory_iterator(folder.path),
                        std::filesystem::directory_iterator{}) == 1, "Failed publish left temp file");
}
void invalid_save_paths() {
    Folder folder;
    io_rejects([&] { save_model_new(make_example_network(), folder.path); });
    io_rejects([&] { save_model_new(make_example_network(), folder.path / "missing" / "model.pgrnn"); });
    io_rejects([&] { save_model_new(make_example_network(), {}); });
    check(!std::filesystem::exists(folder.path / "missing"), "Parent created implicitly");
    check(std::filesystem::is_empty(folder.path), "Invalid save leaked temp file");
}
void invalid_load_paths_and_files() {
    Folder folder;
    io_rejects([&] { static_cast<void>(load_model(folder.path)); });
    io_rejects([&] { static_cast<void>(load_model(folder.path / "missing.pgrnn")); });
    const auto target = folder.path / "bad.pgrnn";
    write_bytes(target, {1, 2, 3});
    io_rejects([&] { static_cast<void>(load_model(target)); });
    auto bad = serialize_model(make_example_network());
    bad.resize(132);
    write_bytes(target, bad);
    io_rejects([&] { static_cast<void>(load_model(target)); });
}
void oversized_disk_file_rejected() {
    Folder folder;
    const auto target = folder.path / "oversized.pgrnn";
    std::ofstream out(target, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.seekp(static_cast<std::streamoff>(max_model_bytes));
    out.put('x');
    out.close();
    io_rejects([&] { static_cast<void>(load_model(target)); });
}
void unicode_and_spaces_paths() {
    Folder folder;
    const auto sub = folder.path / std::filesystem::u8path(u8"\u041e\u043f\u044b\u0442 01");
    check(std::filesystem::create_directory(sub), "Unicode subdirectory failed");
    const auto target = sub / std::filesystem::u8path(u8"\u043c\u043e\u0434\u0435\u043b\u044c.pgrnn");
    const auto source = make_example_network();
    save_model_new(source, target);
    check(serialize_model(load_model(target)) == serialize_model(source), "Unicode path round-trip failed");
}
} // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"external_binary_reference", external_binary_reference},
        {"deterministic_bytes", deterministic_bytes}, {"exact_all_traces", exact_all_traces},
        {"trained_weights_roundtrip", trained_weights_roundtrip},
        {"floating_point_bits_preserved", floating_point_bits_preserved},
        {"arbitrary_class_context_sizes", arbitrary_class_and_context_sizes},
        {"every_truncation_rejected", every_truncation_rejected}, {"extra_bytes_rejected", extra_bytes_rejected},
        {"magic_rejected", magic_rejected}, {"version_rejected", version_rejected},
        {"header_length_rejected", header_length_rejected}, {"scalar_type_rejected", scalar_type_rejected},
        {"flags_rejected", flags_rejected}, {"architecture_rejected", architecture_rejected},
        {"reserved_rejected", reserved_rejected}, {"zero_large_dimensions", zero_and_large_dimensions},
        {"matrix_product_limited", matrix_product_limited_before_allocation},
        {"declared_sizes_rejected", declared_sizes_rejected},
        {"crc_covers_payload_header", crc_covers_payload_and_header},
        {"nan_inf_valid_crc_rejected", nan_inf_rejected_even_with_valid_crc},
        {"malformed_load_keeps_source", malformed_load_does_not_change_source},
        {"file_roundtrip_no_temp", file_roundtrip_and_no_temp_left},
        {"existing_file_preserved", existing_file_preserved}, {"invalid_save_paths", invalid_save_paths},
        {"invalid_load_paths_files", invalid_load_paths_and_files},
        {"oversized_disk_file_rejected", oversized_disk_file_rejected},
        {"unicode_spaces_paths", unicode_and_spaces_paths}
    };
    std::size_t failures = 0;
    for (const auto& test : tests) {
        try { test.second(); std::cout << "PASS " << test.first << '\n'; }
        catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << "Storage tests: " << tests.size() << "; failures: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
