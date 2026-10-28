#include <pgr/neural/model_io.hpp>
#include "file_io.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pgr::neural {
namespace {
// На другой арифметике программу следует остановить при компиляции, а не
// создавать внешне похожий файл с неверным смыслом чисел.
static_assert(sizeof(Real) == 8 && std::numeric_limits<Real>::is_iec559 &&
              std::numeric_limits<Real>::digits == 53 &&
              std::numeric_limits<Real>::max_exponent == 1024,
              "PGRNN v1 requires IEEE-754 binary64 Real");
constexpr std::array<std::uint8_t, 8> magic = {'P', 'G', 'R', 'N', 'N', 13, 10, 26};
constexpr std::uint32_t binary64_type = 1;
constexpr std::uint32_t architecture_id = 1;
constexpr std::size_t checksum_bytes = 4;

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(message);
}

// Побайтовый сдвиг не зависит от выравнивания указателя и endian машины.
void put(ModelBytes& out, std::size_t offset, std::uint64_t value, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        out[offset + i] = static_cast<std::uint8_t>(value & 255U);
        value >>= 8U;
    }
}
std::uint64_t get(const ModelBytes& in, std::size_t offset, std::size_t count) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < count; ++i) {
        value |= static_cast<std::uint64_t>(in[offset + i]) << (i * 8U);
    }
    return value;
}

// CRC-32/ISO-HDLC (IEEE): reflected polynomial 0xEDB88320,
// init=0xFFFFFFFF, xorout=0xFFFFFFFF. Не хеш для проверки подлинности.
std::uint32_t crc32(const ModelBytes& bytes, std::size_t count) {
    std::uint32_t crc = UINT32_C(0xFFFFFFFF);
    for (std::size_t i = 0; i < count; ++i) {
        crc ^= bytes[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            if ((crc & 1U) != 0U) {
                crc = (crc >> 1U) ^ UINT32_C(0xEDB88320);
            } else {
                crc >>= 1U;
            }
        }
    }
    return crc ^ UINT32_C(0xFFFFFFFF);
}

// Все произведения проверяются ДО умножения и выделения vector.
std::size_t matrix_elements(std::size_t rows, std::size_t columns) {
    if (rows == 0 || columns == 0 || rows > Matrix::max_elements / columns) {
        throw std::length_error("PGRNN matrix exceeds element limit");
    } else {
        return rows * columns;
    }
}
struct Layout {
    Shape shape;
    std::size_t count;
    std::size_t total;
};
Layout layout_for(const Shape& s) {
    validate_shape(s);
    const auto e = matrix_elements(s.vocabulary, s.embedding_dimension);
    const auto w1 = matrix_elements(s.hidden_size,
        s.embedding_dimension + s.compiler_count + s.stage_count);
    const auto w2 = matrix_elements(s.class_count, s.hidden_size);
    // После отдельных проверок сумма <= 3*2'000'000+2*4096; размер
    // представим даже в 32-битном size_t. Затем действует лимит всего файла.
    const auto count = e + w1 + s.hidden_size + w2 + s.class_count;
    if (count > (max_model_bytes - model_header_bytes - checksum_bytes) / 8U) {
        throw std::length_error("PGRNN exceeds file byte limit");
    } else {
        return {s, count, model_header_bytes + count * 8U + checksum_bytes};
    }
}

Layout inspect_header(const ModelBytes& bytes) {
    if (bytes.size() < model_header_bytes) {
        invalid("PGRNN header is truncated");
    } else if (!std::equal(magic.begin(), magic.end(), bytes.begin())) {
        invalid("PGRNN magic mismatch");
    } else if (get(bytes, 8, 4) != model_format_version) {
        invalid("Unsupported PGRNN format version");
    } else if (get(bytes, 12, 4) != model_header_bytes ||
               get(bytes, 16, 4) != binary64_type || get(bytes, 20, 4) != 0 ||
               get(bytes, 104, 4) != architecture_id) {
        invalid("Unsupported PGRNN header/type/flags/architecture");
    } else {
        // Расширения формата должны получить собственную версию/правила.
    }
    for (std::size_t i = 108; i < model_header_bytes; ++i) {
        if (bytes[i] != 0) {
            invalid("PGRNN reserved bytes must be zero");
        } else {
            // В v1 скрытых полей в резерве нет.
        }
    }
    std::array<std::size_t, 7> dims{};
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const auto value = get(bytes, 32 + 8 * i, 8);
        if (value > std::numeric_limits<std::size_t>::max()) {
            throw std::length_error("PGRNN dimension does not fit size_t");
        } else {
            dims[i] = static_cast<std::size_t>(value);
        }
    }
    const auto layout = layout_for({dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]});
    if (get(bytes, 24, 8) != layout.total || get(bytes, 88, 8) != layout.count ||
        get(bytes, 96, 8) != layout.count * 8U) {
        invalid("PGRNN declared sizes do not match Shape");
    } else {
        return layout;
    }
}

void append_number(ModelBytes& out, std::size_t& offset, Real value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits)); // Нет нарушения strict aliasing.
    put(out, offset, bits, 8);
    offset += 8;
}
void append_matrix(ModelBytes& out, std::size_t& offset, const Matrix& matrix) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.columns(); ++col) {
            append_number(out, offset, matrix.at(row, col));
        }
    }
}
Vector read_vector(const ModelBytes& bytes, std::size_t& offset, std::size_t count) {
    Vector result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto bits = get(bytes, offset, 8);
        Real value = 0;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) {
            invalid("PGRNN parameter is NaN or infinity");
        } else {
            result.push_back(value);
        }
        offset += 8;
    }
    return result;
}
} // namespace

ModelBytes serialize_model(const Network& network) {
    const auto l = layout_for(network.shape());
    const auto& s = l.shape;
    const auto& p = network.parameters();
    ModelBytes bytes(l.total, 0);
    std::copy(magic.begin(), magic.end(), bytes.begin());
    put(bytes, 8, model_format_version, 4);
    put(bytes, 12, model_header_bytes, 4);
    put(bytes, 16, binary64_type, 4);
    put(bytes, 24, l.total, 8);
    const std::size_t dims[] = {s.vocabulary, s.embedding_dimension, s.compiler_count,
        s.stage_count, s.hidden_size, s.class_count, s.max_tokens};
    for (std::size_t i = 0; i < 7; ++i) {
        put(bytes, 32 + 8 * i, dims[i], 8);
    }
    put(bytes, 88, l.count, 8);
    put(bytes, 96, l.count * 8U, 8);
    put(bytes, 104, architecture_id, 4);
    std::size_t offset = model_header_bytes;
    append_matrix(bytes, offset, p.embeddings);
    append_matrix(bytes, offset, p.w1);
    for (const auto value : p.b1) { append_number(bytes, offset, value); }
    append_matrix(bytes, offset, p.w2);
    for (const auto value : p.b2) { append_number(bytes, offset, value); }
    put(bytes, offset, crc32(bytes, offset), checksum_bytes);
    return bytes;
}

Network deserialize_model(const ModelBytes& bytes) {
    if (bytes.size() > max_model_bytes) {
        throw std::length_error("PGRNN input exceeds file byte limit");
    } else {
        // Вызвавшая сторона уже владеет bytes, но весов ещё не выделено.
    }
    const auto layout = inspect_header(bytes);
    if (bytes.size() != layout.total) {
        invalid("PGRNN file is truncated or contains trailing bytes");
    } else if (get(bytes, bytes.size() - checksum_bytes, checksum_bytes) !=
               crc32(bytes, bytes.size() - checksum_bytes)) {
        invalid("PGRNN CRC-32 mismatch");
    } else {
        // Только теперь создаём массивы параметров. Не используем сырую
        // выгрузку структуры и не принимаем дополнительные невидимые веса.
    }
    const auto& s = layout.shape;
    std::size_t offset = model_header_bytes;
    // Последовательные переменные важны: порядок вычисления аргументов
    // функции в C++17 не задаёт порядок чтения пяти массивов.
    Matrix e(s.vocabulary, s.embedding_dimension,
             read_vector(bytes, offset, s.vocabulary * s.embedding_dimension));
    Matrix w1(s.hidden_size, s.embedding_dimension + s.compiler_count + s.stage_count,
              read_vector(bytes, offset, s.hidden_size *
                  (s.embedding_dimension + s.compiler_count + s.stage_count)));
    auto b1 = read_vector(bytes, offset, s.hidden_size);
    Matrix w2(s.class_count, s.hidden_size,
              read_vector(bytes, offset, s.class_count * s.hidden_size));
    auto b2 = read_vector(bytes, offset, s.class_count);
    return Network(s, {std::move(e), std::move(w1), std::move(b1), std::move(w2), std::move(b2)});
}

Network load_model(const std::filesystem::path& file) {
    if (!std::filesystem::is_regular_file(file)) {
        invalid("PGRNN input must be an existing regular file");
    } else {
        // Устройство/каталог не является источником модели.
    }
    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Cannot open PGRNN input file");
    } else {
        // Размер узнаём на открытом потоке, а не доверяем имени/расширению.
    }
    const auto length = stream.tellg();
    if (length < static_cast<std::streamoff>(model_header_bytes + checksum_bytes) ||
        length > static_cast<std::streamoff>(max_model_bytes)) {
        throw std::length_error("PGRNN input file size is outside limits");
    } else {
        stream.seekg(0);
    }
    ModelBytes bytes(model_header_bytes);
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        invalid("PGRNN header read failed");
    } else {
        // Даже поддельное поле total_bytes не вызывает выделение гигабайт.
    }
    const auto layout = inspect_header(bytes);
    if (length != static_cast<std::streamoff>(layout.total)) {
        invalid("PGRNN disk size does not match header");
    } else {
        bytes.resize(layout.total);
    }
    stream.read(reinterpret_cast<char*>(bytes.data() + model_header_bytes),
                static_cast<std::streamsize>(bytes.size() - model_header_bytes));
    if (!stream) {
        invalid("PGRNN body read failed");
    } else {
        char extra = 0;
        if (stream.get(extra)) {
            invalid("PGRNN file grew during read");
        } else if (!stream.eof() || stream.bad()) {
            throw std::runtime_error("PGRNN end-of-file read failed");
        } else {
            return deserialize_model(bytes);
        }
    }
}

void save_model_new(const Network& network, const std::filesystem::path& file) {
    const auto bytes = serialize_model(network);
    detail::publish_new_file(file, bytes);
}
} // namespace pgr::neural
