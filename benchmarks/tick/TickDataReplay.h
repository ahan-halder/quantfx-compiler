#pragma once

#include <string>
#include <vector>

namespace qfx {
namespace tick {

struct TickMessage {
    double timestamp; // Seconds from midnight
    int type;         // 1: Submission, 2: Cancellation, 3: Deletion, 4: Execution, 5: Hidden Execution
    long long orderId;
    int size;         // Volume
    float price;      // Price
    int direction;    // 1: Buy, -1: Sell
};

class TickDataReplay {
public:
    TickDataReplay(const std::string& filepath);
    
    // Read the CSV file into memory
    bool load();
    
    const std::vector<TickMessage>& getMessages() const { return messages_; }

private:
    std::string filepath_;
    std::vector<TickMessage> messages_;
};

} // namespace tick
} // namespace qfx
