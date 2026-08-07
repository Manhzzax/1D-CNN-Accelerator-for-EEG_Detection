#include "episepset_5k_contract.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace episepset_5k {
void episepset_5k_golden(const std::int16_t input[kChannels][kSamples], std::int64_t logits[2]);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: episepset_5k_golden <test_vectors_directory>\n";
        return 2;
    }
    const std::string directory(argv[1]);
    std::array<std::int16_t, episepset_5k::kChannels * episepset_5k::kSamples> input{};
    std::ifstream input_file(directory + "/input_i16.bin", std::ios::binary);
    input_file.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(input.size() * sizeof(input.front())));
    if (input_file.gcount() != static_cast<std::streamsize>(input.size() * sizeof(input.front()))) {
        std::cerr << "Could not read exactly 17x512 INT16 input values\n";
        return 3;
    }
    std::array<std::int64_t, 2> expected{};
    std::ifstream expected_file(directory + "/expected_logits_i64.txt");
    if (!(expected_file >> expected[0] >> expected[1])) {
        std::cerr << "Could not read two expected INT64 logits\n";
        return 4;
    }

    std::array<std::int64_t, 2> actual{};
    episepset_5k::episepset_5k_golden(
        reinterpret_cast<const std::int16_t (*)[episepset_5k::kSamples]>(input.data()), actual.data());
    std::cout << "expected_logits=" << expected[0] << "," << expected[1] << "\n";
    std::cout << "actual_logits=" << actual[0] << "," << actual[1] << "\n";
    if (actual != expected) {
        std::cerr << "M1a golden-vector mismatch\n";
        return 1;
    }
    std::cout << "M1a golden-vector PASS\n";
    return 0;
}
