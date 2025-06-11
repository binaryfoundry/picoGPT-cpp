// A C++ implementation of GPT-2’s Byte Pair Encoding (BPE) utilities, converted
// from OpenAI’s `encoder.py`. Uses nlohmann::json for JSON parsing and
// <regex> from the C++ STL for tokenization. Header-only aside from nlohmann/json.
//
// To use, include this header, then instantiate `BPEEncoder encoder(models_dir, model_name);`
// and call `encoder.encode(text)` / `encoder.decode(ids)`.
//
// Dependencies (header-only):
//   - nlohmann/json.hpp  (JSON parsing)
//   - <regex>, <fstream>, <sstream>, <unordered_map>, <vector>, <string>
//

#pragma once

#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// -----------------------------------------------------------------------------
// Helper: Convert a Unicode codepoint to a UTF-8 std::string
// -----------------------------------------------------------------------------
static std::string codepoint_to_utf8(int cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// -----------------------------------------------------------------------------
// Helper: Split a UTF-8 string into a vector of individual Unicode codepoint strings.
// -----------------------------------------------------------------------------
static std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> result;
    size_t i = 0, len = s.size();
    while (i < len) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t char_len = 1;
        if ((c & 0x80) == 0x00) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }
        result.emplace_back(s.substr(i, char_len));
        i += char_len;
    }
    return result;
}

// -----------------------------------------------------------------------------
// Helper: Create a reversible mapping from bytes [0..255] to Unicode codepoints,
// and back. Equivalent to Python’s bytes_to_unicode() with caching.
// -----------------------------------------------------------------------------
static std::unordered_map<uint8_t, std::string> build_byte_encoder() {
    // bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    std::vector<int> bs;
    for (int b = int('!'); b <= int('~'); ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b)    bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b)    bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    // cs now contains codepoints; convert to UTF-8 strings
    std::unordered_map<uint8_t, std::string> be;
    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t byte = static_cast<uint8_t>(bs[i]);
        int codep = cs[i];
        be[byte] = codepoint_to_utf8(codep);
    }
    return be;
}

// -----------------------------------------------------------------------------
// Helper: Build reverse of byte_encoder
// -----------------------------------------------------------------------------
static std::unordered_map<std::string, uint8_t> build_byte_decoder(
    const std::unordered_map<uint8_t, std::string>& byte_enc) 
{
    std::unordered_map<std::string, uint8_t> bd;
    for (auto& kv : byte_enc) {
        bd[kv.second] = kv.first;
    }
    return bd;
}

// -----------------------------------------------------------------------------
// Helper: Given a “word” (vector of codepoint‐strings), return all adjacent pairs.
// -----------------------------------------------------------------------------
static std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) return pairs;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
        pairs.insert({ word[i], word[i+1] });
    }
    return pairs;
}

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

// -----------------------------------------------------------------------------
// BPEEncoder: Implements Byte-Pair Encoding (BPE) for GPT-2 in C++.
// -----------------------------------------------------------------------------
class BPEEncoder {
public:
    // Constructor: loads encoder.json and vocab.bpe from disk
    // models_dir/model_name/encoder.json and vocab.bpe
    BPEEncoder(const std::string& models_dir, const std::string& errors = "replace")
        : errors_(errors)
    {
        // 1) Load encoder.json into encoder_ (map<string,int>)
        {
            std::string enc_path = models_dir + "\\encoder.json";
            std::ifstream fin(enc_path);
            if (!fin) {
                throw std::runtime_error("Could not open: " + enc_path);
            }
            json j; 
            fin >> j;
            for (auto it = j.begin(); it != j.end(); ++it) {
                encoder_[it.key()] = it.value().get<int>();
            }
        }

        // 2) Load vocab.bpe (all merges)
        {
            std::string bpe_path = models_dir + "\\vocab.bpe";
            std::ifstream fin(bpe_path);
            if (!fin) {
                throw std::runtime_error("Could not open: " + bpe_path);
            }
            std::string line;
            // First line is a header, skip it
            std::getline(fin, line);
            // Next lines: each "A B"
            std::vector<std::pair<std::string, std::string>> merges;
            while (std::getline(fin, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                std::string a, b;
                iss >> a >> b;
                merges.emplace_back(a, b);
            }
            bpe_merges_ = std::move(merges);
        }

        // 3) Build decoder_ (int → string)
        for (auto& kv : encoder_) {
            decoder_[kv.second] = kv.first;
        }

        // 4) Build byte encoder/decoder
        byte_encoder_ = build_byte_encoder();
        byte_decoder_ = build_byte_decoder(byte_encoder_);

        // 5) Build bpe_ranks_: map<pair<string,string> → int>
        for (size_t i = 0; i < bpe_merges_.size(); ++i) {
            bpe_ranks_[bpe_merges_[i]] = static_cast<int>(i);
        }

        // 6) Compile regex for tokenizing text
        //    Pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+"
        //    (approximation of Python’s Unicode pattern)
        pat_ = std::regex(
            R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+)",
            std::regex::ECMAScript
        );
    }

    // Encode a plain UTF-8 text string into BPE token IDs
    std::vector<int> encode(const std::string& text) {
        std::vector<int> bpe_tokens;
        auto begin = std::sregex_iterator(text.begin(), text.end(), pat_);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string token = it->str();
            // 1) Convert token’s raw bytes to our byte-to-unicode map
            std::string token_u8;
            for (unsigned char c : token) {
                token_u8 += byte_encoder_[c];
            }
            // 2) Apply BPE on token_u8 (returns space-separated BPE pieces)
            std::string token_bpe = bpe(token_u8);
            // 3) Split on space and map each piece to encoder_ ID
            std::istringstream iss(token_bpe);
            std::string piece;
            while (iss >> piece) {
                auto find_it = encoder_.find(piece);
                if (find_it == encoder_.end()) {
                    throw std::runtime_error("Unknown BPE piece: " + piece);
                }
                bpe_tokens.push_back(find_it->second);
            }
        }
        return bpe_tokens;
    }

    // Decode a sequence of BPE token IDs back to plain UTF-8 text
    std::string decode(const std::vector<int>& tokens) {
        // 1) Map each ID → string (unicode)
        std::string text_u8; 
        for (int id : tokens) {
            auto it = decoder_.find(id);
            if (it == decoder_.end()) {
                throw std::runtime_error("Unknown token id: " + std::to_string(id));
            }
            text_u8 += it->second;
        }
        // 2) Split text_u8 into UTF-8 codepoint strings
        auto chars = split_utf8(text_u8);
        // 3) Map each codepoint → original byte
        std::vector<unsigned char> bytes;
        bytes.reserve(chars.size());
        for (auto& cp : chars) {
            auto it = byte_decoder_.find(cp);
            if (it == byte_decoder_.end()) {
                // If error mode is "replace", insert replacement char
                if (errors_ == "replace") {
                    bytes.push_back(static_cast<unsigned char>('?'));
                } else {
                    throw std::runtime_error("Byte-decoding error for: " + cp);
                }
            } else {
                bytes.push_back(it->second);
            }
        }
        // 4) Construct a UTF-8 string from raw bytes
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

private:
    // Apply BPE on a single “token” (already mapped to byte-unicode).
    // Returns a single string where BPE pieces are separated by spaces.
    std::string bpe(const std::string& token) {
        // Check cache
        auto cache_it = cache_.find(token);
        if (cache_it != cache_.end()) {
            return cache_it->second;
        }

        // 1) Split token into “characters” (UTF-8 codepoints)
        std::vector<std::string> word = split_utf8(token);
        auto pairs = get_pairs(word);
        if (pairs.empty()) {
            cache_[token] = token;
            return token;
        }

        // 2) Repeatedly merge the highest-ranking bigram
        while (true) {
            // Find bigram with lowest rank (smallest index) among current pairs
            int best_rank = INT_MAX;
            std::pair<std::string, std::string> best_bigram;
            bool found = false;
            for (auto& p : pairs) {
                auto rank_it = bpe_ranks_.find(p);
                if (rank_it != bpe_ranks_.end() && rank_it->second < best_rank) {
                    best_rank = rank_it->second;
                    best_bigram = p;
                    found = true;
                }
            }
            if (!found) break;  // no more merges found in ranks

            // Merge occurrences of best_bigram in “word”
            std::vector<std::string> new_word;
            new_word.reserve(word.size());
            size_t i = 0;
            while (i < word.size()) {
                if (i + 1 < word.size() &&
                    word[i] == best_bigram.first &&
                    word[i+1] == best_bigram.second)
                {
                    // Merge them
                    new_word.push_back(word[i] + word[i+1]);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i += 1;
                }
            }
            word = std::move(new_word);
            if (word.size() == 1) break;
            pairs = get_pairs(word);
        }

        // 3) Join word pieces with spaces
        std::ostringstream oss;
        for (size_t i = 0; i < word.size(); ++i) {
            if (i) oss << ' ';
            oss << word[i];
        }
        std::string result = oss.str();
        cache_[token] = result;
        return result;
    }

    // Members
    std::unordered_map<std::string, int> encoder_;               // token → ID
    std::unordered_map<int, std::string> decoder_;               // ID → token
    std::vector<std::pair<std::string, std::string>> bpe_merges_; // list of merges
    std::unordered_map<std::pair<std::string, std::string>, int, PairStringHash, PairStringEqual> bpe_ranks_;
    std::unordered_map<std::string, std::string> cache_;         // token → BPE'd string
    std::unordered_map<uint8_t, std::string> byte_encoder_;      // byte → unicode string
    std::unordered_map<std::string, uint8_t> byte_decoder_;      // unicode string → byte
    std::regex pat_;                                             // tokenization regex
    std::string errors_;                                         // how to handle decoding errors
};

