// C++ implementation of GPT-2 inference (picoGPT style) plus
// loading of model weights and metadata (as produced by the accompanying
// Python script). Requires a single‐header JSON library (e.g. nlohmann/json.hpp).
//
// On Windows run setup-env.bat to set up the Python environment required to download the
// GPT2 Tensorflow checkpoint.
// 
// Run fetch-gpt2.py which will convert the checkpoint to JSON which is loaded in here.
//
// ------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

// Include the nlohmann::json single‐header library.
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "encoder.hpp"

using std::ifstream;
using std::ofstream;
using std::size_t;
using std::string;
using std::vector;
using std::unordered_map;

// A simple alias for a 2D float matrix (row-major).
using Matrix = vector<vector<float>>;

// ------------------------------------------------------------
//  Activation Functions / Normalization / Linear Layer
// ------------------------------------------------------------

// GELU activation (tanh approximation)
inline float gelu(const float x) {
    static const float sqrt2_over_pi = std::sqrt(2.0f / 3.14159265358979323846f);
    static const float coeff         = 0.044715f;
    const float x_cubed = x * x * x;
    const float inner   = sqrt2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// Softmax over a single vector (numerically stable, row-wise)
vector<float> softmax(const vector<float>& input) {
    const size_t dim = input.size();
    vector<float> output(dim);
    const float max_val = *std::max_element(input.begin(), input.end());
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double e = std::exp(input[i] - max_val);
        output[i] = static_cast<float>(e);
        sum += e;
    }
    for (size_t i = 0; i < dim; ++i) {
        output[i] = static_cast<float>(output[i] / sum);
    }
    return output;
}

// Layer Normalization over the last dimension of each row in X.
// X: [n_seq x dim], gamma and beta are length dim.
Matrix layer_norm(const Matrix& X,
                  const vector<float>& gamma,
                  const vector<float>& beta,
                  const float eps = 1e-5f)
{
    const int n_seq = static_cast<int>(X.size());
    const int dim   = static_cast<int>(X[0].size());
    Matrix Y(n_seq, vector<float>(dim, 0.0f));

    for (int i = 0; i < n_seq; ++i) {
        // Compute mean
        double mean = 0.0;
        for (int j = 0; j < dim; ++j) {
            mean += X[i][j];
        }
        mean /= dim;

        // Compute variance
        double var = 0.0;
        for (int j = 0; j < dim; ++j) {
            const double diff = X[i][j] - mean;
            var += diff * diff;
        }
        var /= dim;

        const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);

        // Normalize + scale + shift
        for (int j = 0; j < dim; ++j) {
            const float normalized = (X[i][j] - static_cast<float>(mean)) * inv_std;
            Y[i][j] = gamma[j] * normalized + beta[j];
        }
    }
    return Y;
}

// Linear layer: Y = X * W + b, where
//   X: [m x in_dim]
//   W: [in_dim x out_dim]
//   b: [out_dim]
// returns Y: [m x out_dim]
Matrix linear(const Matrix& X,
              const Matrix& W,
              const vector<float>& b)
{
    const int m       = static_cast<int>(X.size());
    const int in_dim  = static_cast<int>(W.size());
    const int out_dim = static_cast<int>(W[0].size());
    Matrix Y(m, vector<float>(out_dim, 0.0f));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            float sum = b[j];
            for (int k = 0; k < in_dim; ++k) {
                sum += X[i][k] * W[k][j];
            }
            Y[i][j] = sum;
        }
    }
    return Y;
}

// ------------------------------------------------------------
//  Attention and Multi-Head Attention
// ------------------------------------------------------------

// Scaled dot-product attention with causal mask.
// Q: [n_q x d_k], K: [n_k x d_k], V: [n_k x d_v], mask: [n_q x n_k]
// Returns: [n_q x d_v]
Matrix attention(const Matrix& Q,
                 const Matrix& K,
                 const Matrix& V,
                 const Matrix& mask)
{
    const int n_q = static_cast<int>(Q.size());
    const int n_k = static_cast<int>(K.size());
    const int d_k = static_cast<int>(Q[0].size());
    const int d_v = static_cast<int>(V[0].size());

    // Compute S = Q * K^T / sqrt(d_k) + mask
    Matrix S(n_q, vector<float>(n_k, 0.0f));
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    for (int i = 0; i < n_q; ++i) {
        for (int j = 0; j < n_k; ++j) {
            double dot = 0.0;
            for (int t = 0; t < d_k; ++t) {
                dot += static_cast<double>(Q[i][t]) * static_cast<double>(K[j][t]);
            }
            S[i][j] = static_cast<float>(dot * scale) + mask[i][j];
        }
    }

    // Apply softmax row-wise
    Matrix P(n_q, vector<float>(n_k, 0.0f));
    for (int i = 0; i < n_q; ++i) {
        P[i] = softmax(S[i]);
    }

    // Compute P * V → [n_q x d_v]
    Matrix out(n_q, vector<float>(d_v, 0.0f));
    for (int i = 0; i < n_q; ++i) {
        for (int j = 0; j < d_v; ++j) {
            double sum = 0.0;
            for (int t = 0; t < n_k; ++t) {
                sum += static_cast<double>(P[i][t]) * static_cast<double>(V[t][j]);
            }
            out[i][j] = static_cast<float>(sum);
        }
    }
    return out;
}

// Multi-Head Attention (MHA) with causal masking.
// X: [n_seq x n_embd]
// Wqkv: [n_embd x (3 * n_embd)], bqkv: [(3 * n_embd)]
// Wproj: [n_embd x n_embd], bproj: [n_embd]
// n_head: number of attention heads
Matrix mha(const Matrix& X,
           const Matrix& Wqkv,
           const vector<float>& bqkv,
           const Matrix& Wproj,
           const vector<float>& bproj,
           const int n_head)
{
    const int n_seq  = static_cast<int>(X.size());
    const int n_embd = static_cast<int>(X[0].size());
    const int d_model = n_embd;

    // 1) Project X → QKV combined: [n_seq x 3*n_embd]
    const Matrix qkv_combined = linear(X, Wqkv, bqkv);

    // 2) Split into Q, K, V: each [n_seq x n_embd]
    Matrix Q(n_seq, vector<float>(n_embd));
    Matrix K(n_seq, vector<float>(n_embd));
    Matrix V(n_seq, vector<float>(n_embd));
    for (int i = 0; i < n_seq; ++i) {
        for (int j = 0; j < n_embd; ++j) {
            Q[i][j] = qkv_combined[i][j];
            K[i][j] = qkv_combined[i][n_embd + j];
            V[i][j] = qkv_combined[i][2 * n_embd + j];
        }
    } 

    // 3) Create causal mask [n_seq x n_seq]: mask[i][j] = 0 if j ≤ i, else -1e10
    Matrix mask(n_seq, vector<float>(n_seq, 0.0f));
    for (int i = 0; i < n_seq; ++i) {
        for (int j = 0; j < n_seq; ++j) {
            mask[i][j] = (j > i ? -1e10f : 0.0f);
        }
    }

    // 4) Split Q,K,V into heads. Each head has dimension d_head = n_embd / n_head
    const int d_head = d_model / n_head;
    vector<Matrix> head_outputs;
    head_outputs.reserve(n_head);

    for (int h = 0; h < n_head; ++h) {
        // Extract head‐slice for Q, K, V
        Matrix Qh(n_seq, vector<float>(d_head));
        Matrix Kh(n_seq, vector<float>(d_head));
        Matrix Vh(n_seq, vector<float>(d_head));
        const int start = h * d_head;
        for (int i = 0; i < n_seq; ++i) {
            for (int j = 0; j < d_head; ++j) {
                Qh[i][j] = Q[i][start + j];
                Kh[i][j] = K[i][start + j];
                Vh[i][j] = V[i][start + j];
            }
        }
        // Apply scaled dot-product attention for this head 
        head_outputs.push_back(attention(Qh, Kh, Vh, mask));  // [n_seq x d_head]
    }

    // 5) Concatenate all head outputs → [n_seq x n_embd]
    Matrix concat(n_seq, vector<float>(n_embd, 0.0f));
    for (int i = 0; i < n_seq; ++i) {
        for (int h = 0; h < n_head; ++h) {
            const int start = h * d_head;
            for (int j = 0; j < d_head; ++j) {
                concat[i][start + j] = head_outputs[h][i][j];
            }
        }
    }

    // 6) Final linear projection: [n_seq x n_embd] → [n_seq x n_embd]
    return linear(concat, Wproj, bproj);
}

// ------------------------------------------------------------
//  Transformer Block
// ------------------------------------------------------------

// Struct to hold one block's weights
struct BlockWeights {
    // LayerNorm 1 (pre-attention)
    vector<float> ln1_gamma;  // size: n_embd
    vector<float> ln1_beta;   // size: n_embd
    // Attention sub-layer
    Matrix Wqkv;              // [n_embd x 3*n_embd]
    vector<float> bqkv;       // [3*n_embd]
    Matrix Wproj;             // [n_embd x n_embd]
    vector<float> bproj;      // [n_embd]
    // LayerNorm 2 (pre-FFN)
    vector<float> ln2_gamma;  // [n_embd]
    vector<float> ln2_beta;   // [n_embd]
    // Feed-Forward sub-layer
    Matrix W_fc;              // [n_embd x (4*n_embd)]
    vector<float> b_fc;       // [4*n_embd]
    Matrix W_proj;            // [(4*n_embd) x n_embd]
    vector<float> b_proj;     // [n_embd]
};

// Single transformer block forward pass (GPT-2 style, pre‐norm + residuals):
// x_in: [n_seq x n_embd] → x_out: [n_seq x n_embd]
Matrix transformer_block(const Matrix& x_in,
                         const BlockWeights& W,
                         const int n_head)
{
    Matrix x = x_in;  // copy for residual

    // 1) LayerNorm → MHA → Residual
    Matrix norm1 = layer_norm(x, W.ln1_gamma, W.ln1_beta);
    const Matrix attn_out = mha(norm1,
                                W.Wqkv, W.bqkv,
                                W.Wproj, W.bproj,
                                n_head);
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            x[i][j] += attn_out[i][j];
        }
    }

    // 2) LayerNorm → FFN → Residual
    Matrix norm2 = layer_norm(x, W.ln2_gamma, W.ln2_beta);
    // FFN: Linear → GELU → Linear
    Matrix ffn_inner = linear(norm2, W.W_fc, W.b_fc);  // [n_seq x 4*n_embd]
    // Apply GELU elementwise
    for (size_t i = 0; i < ffn_inner.size(); ++i) {
        for (size_t j = 0; j < ffn_inner[i].size(); ++j) {
            ffn_inner[i][j] = gelu(ffn_inner[i][j]);
        }
    }
    const Matrix ffn_out = linear(ffn_inner, W.W_proj, W.b_proj);  // [n_seq x n_embd]
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            x[i][j] += ffn_out[i][j];
        }
    }

    return x;
}

// ------------------------------------------------------------
//  GPT-2 Forward Pass
// ------------------------------------------------------------

// Model container
struct GPT2Model {
    // Embedding layers
    Matrix Wte;  // [n_vocab x n_embd]
    Matrix Wpe;  // [n_ctx x n_embd]

    // Transformer blocks
    vector<BlockWeights> blocks;  // size: n_layer

    // Final layer norm
    vector<float> ln_f_gamma;  // [n_embd]
    vector<float> ln_f_beta;   // [n_embd]

    // Hyperparameters
    int n_embd;
    int n_layer;
    int n_head;
    int n_ctx;
    int n_vocab;
};

// Forward pass: inputs: vector<int> token IDs (length n_seq).
// Returns logits: [n_seq x n_vocab]
Matrix gpt2_forward(const vector<int>& input_tokens,
                    const GPT2Model& model)
{
    const int n_seq    = static_cast<int>(input_tokens.size());
    const int n_embd   = model.n_embd;
    const int n_vocab  = model.n_vocab;

    // 1) Token + positional embeddings: x = Wte[input_id] + Wpe[position]
    Matrix x(n_seq, vector<float>(n_embd, 0.0f));
    for (int i = 0; i < n_seq; ++i) {
        const int tid = input_tokens[i];
        if (tid < 0 || tid >= model.n_vocab) {
            throw std::out_of_range("Token ID out of range in input_tokens");
        }
        for (int j = 0; j < n_embd; ++j) {
            x[i][j] = model.Wte[tid][j] + model.Wpe[i][j];
        }
    }

    // 2) Transformer blocks
    for (int layer = 0; layer < model.n_layer; ++layer) {
        x = transformer_block(x, model.blocks[layer], model.n_head);
    }

    // 3) Final layer norm
    x = layer_norm(x, model.ln_f_gamma, model.ln_f_beta);

    // 4) Output logits: x * Wte^T  → [n_seq x n_vocab]
    Matrix logits(n_seq, vector<float>(n_vocab, 0.0f));
    for (int i = 0; i < n_seq; ++i) {
        for (int v = 0; v < n_vocab; ++v) {
            double sum = 0.0;
            for (int j = 0; j < n_embd; ++j) {
                sum += static_cast<double>(x[i][j]) * static_cast<double>(model.Wte[v][j]);
            }
            logits[i][v] = static_cast<float>(sum);
        }
    }
    return logits;
}

// ------------------------------------------------------------
//  Utility: Read JSON from file
// ------------------------------------------------------------

json read_json(const string& path) {
    ifstream fin(path);
    if (!fin || !fin.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + path);
    }
    json j;
    fin >> j;
    return j;
}

// ------------------------------------------------------------
//  Utility: Read raw float32 binary into a vector<float>
// ------------------------------------------------------------

void read_floats_from_bin(const string& path,
                          vector<float>& buffer,
                          const size_t expected_count)
{
    ifstream fin(path, std::ios::binary);
    if (!fin || !fin.is_open()) {
        throw std::runtime_error("Could not open binary file: " + path);
    }
    buffer.resize(expected_count);
    fin.read(reinterpret_cast<char*>(buffer.data()),
             static_cast<std::streamsize>(expected_count * sizeof(float)));
    if (!fin) {
        throw std::runtime_error("Failed to read expected float count from: " + path);
    }
}

// ------------------------------------------------------------
//  Load Model Weights from disk
//    Expects:
//      <model_dir>/hparams.json
//      <model_dir>/metadata.json
//      <model_dir>/<param_name>.bin  (for every name in metadata.json)
//      <model_dir>/encoder.json, <model_dir>/vocab.bpe  (for tokenization, not handled here)
// ------------------------------------------------------------

GPT2Model load_model(const string& model_dir) {
    // 1) Read hparams.json
    const string hparams_path = model_dir + "/hparams.json";
    const json hps = read_json(hparams_path);

    GPT2Model model;
    model.n_embd     = hps.at("n_embd").get<int>();
    model.n_layer    = hps.at("n_layer").get<int>();
    model.n_head     = hps.at("n_head").get<int>();
    model.n_ctx      = hps.at("n_ctx").get<int>();
    model.n_vocab    = hps.at("n_vocab").get<int>();

    // 2) Read metadata.json
    const string meta_path = model_dir + "/metadata.json";
    const json meta = read_json(meta_path);

    // Helper lambda to fetch shape from metadata
    const auto get_shape = [&](const string& key) -> vector<int> {
        if (!meta.contains(key)) {
            throw std::runtime_error("metadata.json missing key: " + key);
        }
        return meta.at(key).get<vector<int>>();
    };

    // Helper lambda to load a binary file into a Matrix
    const auto load_matrix = [&](const string& key) -> Matrix {
        const vector<int> shape = get_shape(key);
        if (shape.size() != 2) {
            throw std::runtime_error("Expected 2D shape for key: " + key);
        }
        const int dim0 = shape[0];
        const int dim1 = shape[1];
        const size_t total = static_cast<size_t>(dim0) * static_cast<size_t>(dim1);
        vector<float> flat;
        const string bin_path = model_dir + "/" + key + ".bin";
        read_floats_from_bin(bin_path, flat, total);

        // Reshape flat → Matrix
        Matrix M(dim0, vector<float>(dim1));
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                M[i][j] = flat[static_cast<size_t>(i) * dim1 + j];
            }
        }
        return M;
    };

    // Helper lambda to load a binary file into a vector<float>
    const auto load_vector = [&](const string& key) -> vector<float> {
        const vector<int> shape = get_shape(key);
        if (shape.size() != 1) {
            throw std::runtime_error("Expected 1D shape for key: " + key);
        }
        const int dim0 = shape[0];
        const size_t total = static_cast<size_t>(dim0);
        vector<float> vec;
        const string bin_path = model_dir + "/" + key + ".bin";
        read_floats_from_bin(bin_path, vec, total);
        return vec;
    };

    // 3) Load token embedding (WTE) and positional embedding (WPE)
    model.Wte = load_matrix("wte");  // [n_vocab x n_embd]
    model.Wpe = load_matrix("wpe");  // [n_ctx x n_embd]

    // 4) Load final layer norm parameters
    model.ln_f_gamma = load_vector("ln_f_g");  // [n_embd]
    model.ln_f_beta  = load_vector("ln_f_b");  // [n_embd]

    // 5) Resize blocks
    model.blocks.resize(model.n_layer);

    // 6) For each transformer block, load its parameters
    for (int layer = 0; layer < model.n_layer; ++layer) {
        BlockWeights& W = model.blocks[layer];
        // LayerNorm 1
        W.ln1_gamma = load_vector("blocks_" + std::to_string(layer) + "_ln_1_g");
        W.ln1_beta  = load_vector("blocks_" + std::to_string(layer) + "_ln_1_b");
        // Attention: c_attn → Wqkv, bqkv
        W.Wqkv      = load_matrix("blocks_" + std::to_string(layer) + "_attn_c_attn_w");
        W.bqkv      = load_vector("blocks_" + std::to_string(layer) + "_attn_c_attn_b");
        // Attention: c_proj → Wproj, bproj
        W.Wproj     = load_matrix("blocks_" + std::to_string(layer) + "_attn_c_proj_w");
        W.bproj     = load_vector("blocks_" + std::to_string(layer) + "_attn_c_proj_b");
        // LayerNorm 2
        W.ln2_gamma = load_vector("blocks_" + std::to_string(layer) + "_ln_2_g");
        W.ln2_beta  = load_vector("blocks_" + std::to_string(layer) + "_ln_2_b");
        // Feed-Forward: c_fc → W_fc, b_fc
        W.W_fc      = load_matrix("blocks_" + std::to_string(layer) + "_mlp_c_fc_w");
        W.b_fc      = load_vector("blocks_" + std::to_string(layer) + "_mlp_c_fc_b");
        // Feed-Forward: c_proj → W_proj, b_proj
        W.W_proj    = load_matrix("blocks_" + std::to_string(layer) + "_mlp_c_proj_w");
        W.b_proj    = load_vector("blocks_" + std::to_string(layer) + "_mlp_c_proj_b");
    }

    return model;
}

// ------------------------------------------------------------
//  Main Function: Load model, tokenize input (stub), run inference
// ------------------------------------------------------------

int main(const int argc, char* const argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <prompt>\n";
        return 1;
    }
    const string model_dir = argv[1];
    const string prompt    = argv[2];

    std::cout << "Loading model from: " << model_dir << " \n";
    const GPT2Model model = load_model(model_dir);
    std::cout << "Loaded GPT-2 with:\n"
              << "    n_layer    = " << model.n_layer   << "\n"
              << "    n_embd     = " << model.n_embd    << "\n"
              << "    n_head     = " << model.n_head    << "\n"
              << "    n_ctx      = " << model.n_ctx     << "\n"
              << "    n_vocab    = " << model.n_vocab   << "\n\n";

    std::cout << "Prompt:\n" << prompt << "\n\n";

    // --------------------------------------------------------------------
    // Tokenization step (BPE)
    // --------------------------------------------------------------------
    BPEEncoder encoder(model_dir);
    vector<int> input_tokens = encoder.encode(prompt);

    // --------------------------------------------------------------------
    // Run a single forward pass
    // --------------------------------------------------------------------
    const Matrix logits = gpt2_forward(input_tokens, model);

    // --------------------------------------------------------------------
    // Greedy decode: take the logits of the last position
    // --------------------------------------------------------------------
    int last_idx = static_cast<int>(input_tokens.size()) - 1;
    vector<float> last_logits = logits[last_idx];      // shape: [n_vocab]

    // Apply softmax to convert to probabilities
    vector<float> probs = softmax(last_logits);

    // Build (prob, id) pairs and sort descending
    vector<std::pair<float, int>> pid;
    pid.reserve(probs.size());
    for (int i = 0; i < model.n_vocab; ++i) {
        pid.emplace_back(probs[i], i);
    }
    std::sort(pid.begin(), pid.end(),
        [](auto& a, auto& b) { return a.first > b.first; });

    // Print top 10
    std::cout << "Top 10 next tokens:\n";
    for (int i = 0; i < 10 && i < (int)pid.size(); ++i) {
        float p = pid[i].first;
        int   id = pid[i].second;
        // decode single token ID to string
        string tok = encoder.decode(vector<int>{id});
        std::cout
            << std::setw(2) << (i + 1) << ": "
            << "ID=" << id
            << "  prob=" << p
            << "  token=\"" << tok << "\"\n";
    }

    // Find the argmax token ID
    auto max_it = std::max_element(probs.begin(), probs.end());
    int next_id = static_cast<int>(std::distance(probs.begin(), max_it));

    std::cout << "Predicted next token ID: " << next_id
        << "  (prob=" << *max_it << ")\n";
    std::cout << "\n";

    // --------------------------------------------------------------------
    // Append the predicted ID and decode the full sequence back to text
    // --------------------------------------------------------------------
    input_tokens.push_back(next_id);
    string output_text = encoder.decode(input_tokens);

    std::cout << "Decoded text:\n" << output_text << "\n";

    return 0;
}
