#pragma once

#include <pgr/neural/network.hpp>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace pgr::neural {

using ModelBytes = std::vector<std::uint8_t>;

/**
 * @brief Ограничения формата PGRNN v1, независимые от версии программы.
 *
 * Сериализация всегда использует little-endian и IEEE-754 binary64.
 * Лимит файла проверяется до выделения его буфера. Лимиты Shape/Matrix
 * дополнительно проверяются до выделения массивов параметров.
 */
inline constexpr std::uint32_t model_format_version = 1;
inline constexpr std::size_t model_header_bytes = 128;
inline constexpr std::size_t max_model_bytes = 64U * 1024U * 1024U;

/**
 * @brief Shape + E/W1/b1/W2/b2 с CRC-32 всего заголовка и параметров.
 *
 * Не записывает сырую память структуры: padding, size_t и порядок байтов
 * машины не являются частью формата. Одинаковые биты параметров дают
 * одинаковый файл: времени, случайных полей и путей внутри него нет.
 * CRC обнаруживает случайное повреждение, но не удостоверяет автора файла.
 */
ModelBytes serialize_model(const Network& network);

/**
 * @brief Создать новую неизменяемую Network только из полностью верных данных.
 *
 * Отклоняет неизвестную версию/архитектуру/тип, ненулевые reserved/flags,
 * несовместимые размеры, обрезанный файл, лишний хвост, неверный CRC, NaN/Inf.
 * Входной буфер принадлежит вызывающей стороне; функция его не меняет.
 */
Network deserialize_model(const ModelBytes& bytes);

/**
 * @brief Прочитать обычный файл; до полного чтения проверить заголовок/размер.
 *
 * При ошибке бросает исключение, не подставляет случайные/нулевые веса.
 * Принимает std::filesystem::path: на Windows API поддерживает Unicode-путь.
 */
Network load_model(const std::filesystem::path& file);

/**
 * @brief Сохранить полностью подготовленный файл только под НОВЫМ именем.
 *
 * Временный файл создаётся эксклюзивно в том же каталоге, записывается,
 * flush/close, затем публикуется без замены существующей цели. Даже при
 * конкурентном сохранении другой процесс не должен потерять свой файл.
 * Родитель должен существовать. Перезапись и автосоздание каталогов отсутствуют.
 * Это не обещание устойчивости к отключению питания/сбою сетевой ФС.
 */
void save_model_new(const Network& network, const std::filesystem::path& file);

} // namespace pgr::neural
