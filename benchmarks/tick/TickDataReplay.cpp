#include "TickDataReplay.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

namespace qfx {
namespace tick {

TickDataReplay::TickDataReplay(const std::string& filepath) : filepath_(filepath) {}

bool TickDataReplay::load() {
    std::ifstream file(filepath_);
    if (!file.is_open()) {
        std::cerr << "Failed to open tick data file: " << filepath_ << std::endl;
        return false;
    }

    std::string line;
    // Reserve memory if possible, though we don't know the exact size.
    messages_.reserve(1000000);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        TickMessage msg;
        const char* ptr = line.c_str();
        char* endptr = nullptr;

        msg.timestamp = std::strtod(ptr, &endptr);
        if (*endptr == ',') endptr++;
        ptr = endptr;

        msg.type = std::strtol(ptr, &endptr, 10);
        if (*endptr == ',') endptr++;
        ptr = endptr;

        msg.orderId = std::strtoll(ptr, &endptr, 10);
        if (*endptr == ',') endptr++;
        ptr = endptr;

        msg.size = std::strtol(ptr, &endptr, 10);
        if (*endptr == ',') endptr++;
        ptr = endptr;

        msg.price = std::strtof(ptr, &endptr);
        if (*endptr == ',') endptr++;
        ptr = endptr;

        msg.direction = std::strtol(ptr, &endptr, 10);

        messages_.push_back(msg);
    }

    return true;
}

} // namespace tick
} // namespace qfx
