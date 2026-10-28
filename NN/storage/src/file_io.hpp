#pragma once
#include <pgr/neural/model_io.hpp>

namespace pgr::neural::detail {
// Только транспорт байтов. Формат и математические размеры проверяет model_io.
void publish_new_file(const std::filesystem::path& file, const ModelBytes& bytes);
}
