// A C++ implementation of GPT-2's Byte Pair Encoding (BPE) utilities, converted
// from OpenAI’s 'encoder.py'. Uses nlohmann::json for JSON parsing and
// <regex> from the C++ STL for tokenization. Header-only aside from nlohmann/json.

#pragma once

#include <unordered_map>
#include <vector>
#include <regex>
#include <string>

// -----------------------------------------------------------------------------
// Custom hash for std::pair<string,string> so we can use it in unordered_map.
// -----------------------------------------------------------------------------
struct PairStringHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        // A simple combination: hash(first) * 31 + hash(second)
        return std::hash<std::string>()(p.first) * 31u + std::hash<std::string>()(p.second);
    }
};
struct PairStringEqual {
    bool operator()(const std::pair<std::string, std::string>& a,
        const std::pair<std::string, std::string>& b) const noexcept {
        return a.first == b.first && a.second == b.second;
    }
};

class BPEEncoder {
public:
    // Constructor: loads encoder.json and vocab.bpe from disk
    BPEEncoder(const std::string& models_dir, const std::string& errors = "replace");

    // Encode a plain UTF-8 text string into BPE token IDs
    std::vector<int> encode(const std::string& text);

    // Decode a sequence of BPE token IDs back to plain UTF-8 text
    std::string decode(const std::vector<int>& tokens);

private:
    std::string bpe(const std::string& token);

    // Members
    std::unordered_map<std::string, int> encoder_;               // token => ID
    std::unordered_map<int, std::string> decoder_;               // ID => token
    std::vector<std::pair<std::string, std::string>> bpe_merges_; // list of merges
    std::unordered_map<std::pair<std::string, std::string>, int, PairStringHash, PairStringEqual> bpe_ranks_;
    std::unordered_map<std::string, std::string> cache_;         // token => BPE'd string
    std::unordered_map<uint8_t, std::string> byte_encoder_;      // byte => unicode string
    std::unordered_map<std::string, uint8_t> byte_decoder_;      // unicode string => byte
    std::regex pat_;                                             // tokenization regex
    std::string errors_;                                         // how to handle decoding errors
};
;
