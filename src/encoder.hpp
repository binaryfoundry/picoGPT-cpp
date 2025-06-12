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

/**
 * @class BPEEncoder
 * @brief Implements GPT-2 Byte Pair Encoding (BPE) utilities in C++.
 *
 * The BPEEncoder class provides methods to encode and decode text using
 * the Byte Pair Encoding algorithm as used in OpenAI's GPT-2 models.
 * It loads vocabulary and merge rules from model files, tokenizes input
 * text, applies BPE merges, and maps between text and token IDs.
 *
 * Key features:
 * - Loads encoder and BPE merge data from disk.
 * - Encodes UTF-8 strings into sequences of BPE token IDs.
 * - Decodes sequences of BPE token IDs back into UTF-8 strings.
 * - Handles tokenization using regular expressions.
 * - Supports error handling modes for decoding.
 *
 * Dependencies:
 * - nlohmann::json for JSON parsing.
 * - <regex> from the C++ STL for tokenization.
 *
 * Example usage:
 * @code
 *   BPEEncoder encoder("path/to/model/dir");
 *   std::vector<int> tokens = encoder.encode("Hello, world!");
 *   std::string text = encoder.decode(tokens);
 * @endcode
 */
class BPEEncoder {
public:
    /**
     * @brief Constructs a BPEEncoder and loads model files from disk.
     *
     * Loads 'encoder.json' and 'vocab.bpe' from the specified model directory.
     * Initializes the encoder, decoder, BPE merge rules, and byte mappings.
     *
     * @param model_dir  Path to the directory containing model files.
     * @param errors     Error handling mode for decoding ("replace" by default).
     *                   If set to "replace", unknown bytes are replaced with '?'.
     * @throws std::runtime_error if model files cannot be loaded or parsed.
     */
    BPEEncoder(const std::string& model_dir, const std::string& errors = "replace");

    /**
     * @brief Encodes a UTF-8 text string into GPT-2 BPE token IDs.
     *
     * The encoding process is:
     * 1. Uses a regular expression to split the input text into tokens (words,
     *    punctuation, whitespace, contractions).
     * 2. Converts each token's raw bytes into a Unicode string via the byte_encoder_ map.
     * 3. Applies the BPE merge algorithm to the Unicode string, producing space-separated
     *    subword pieces.
     * 4. Splits the BPE result on spaces and maps each piece to its integer ID using encoder_.
     *
     * @param text  The input UTF-8 string to encode.
     * @return      A vector of BPE token IDs corresponding to the input text.
     * @throws std::runtime_error if any BPE piece is not found in the encoder map.
     */
    std::vector<int> encode(const std::string& text);

    /**
     * @brief Decodes a sequence of BPE token IDs back into a UTF-8 string.
     *
     * The decoding process is as follows:
     * 1. Map each token ID to its corresponding Unicode string piece.
     * 2. Concatenate all pieces into a single intermediate UTF-8 string.
     * 3. Split the intermediate string into individual Unicode codepoints.
     * 4. Map each codepoint back to its original byte value, handling unknown
     *    codepoints according to the error mode ("replace" inserts '?').
     * 5. Reassemble the raw bytes into the final UTF-8 output string.
     *
     * @param tokens  Vector of BPE token IDs to decode.
     * @return        The decoded UTF-8 string.
     * @throws std::runtime_error if a token ID is not found in the decoder map
     *                            or if byte decoding fails (unless errors_ == "replace").
     */
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
