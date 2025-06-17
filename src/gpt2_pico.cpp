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

#include "encoder.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using std::ifstream;
using std::ofstream;
using std::size_t;
using std::string;
using std::vector;

// A simple alias for a 2D float matrix (row-major).
using Matrix = vector<vector<float>>;

// ------------------------------------------------------------
//  Activation Functions / Normalization / Linear Layer
// ------------------------------------------------------------

/**
 * @brief Computes the Gaussian Error Linear Unit (GELU) activation.
 *
 * The GELU activation is defined as:
 *     GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * which provides a smoother alternative to the standard ReLU,
 * improving convergence in transformer-based models.
 *
 * @param x The input value.
 * @return The GELU-activated output.
 */
inline float gelu(const float x) {
    static const float sqrt2_over_pi = std::sqrt(2.0f / 3.14159265358979323846f);
    static const float coeff         = 0.044715f;
    const float x_cubed = x * x * x;
    const float inner   = sqrt2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

/**
 * @brief Computes the softmax of a single vector in a numerically stable way.
 *
 * This function subtracts the maximum element from each input value to
 * prevent numerical overflow, exponentiates the shifted values, and then
 * normalizes by the sum of the exponentials so that the returned vector
 * forms a valid probability distribution (sums to 1).
 *
 * @param input A const reference to a vector of input values (logits).
 * @return A vector of the same size as @p input, where each element is
 *         exp(input[i] - max(input)) / sum_j exp(input[j] - max(input)).
 */
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

/**
 * @brief Applies layer normalization to each row of the input matrix.
 *
 * For each row in X (of length dim), this function:
 *  1. Computes the mean of the row.
 *  2. Computes the variance of the row.
 *  3. Normalizes each element: (x - mean) / sqrt(variance + eps).
 *  4. Scales and shifts the normalized values using gamma (scale) and beta (shift).
 *
 * Layer normalization helps stabilize and accelerate training in deep networks
 * by ensuring activations have zero mean and unit variance per feature vector.
 *
 * @param X     Input matrix of shape [n_seq x dim], where n_seq is the number
 *              of elements (e.g., sequence length) and dim is the hidden size.
 * @param gamma Scale (gain) parameters, a vector of length dim.
 * @param beta  Shift (bias) parameters, a vector of length dim.
 * @param eps   Small epsilon value to avoid division by zero when computing
 *              the inverse standard deviation (default: 1e-5f).
 * @return      A matrix of the same shape [n_seq x dim] containing the
 *              layer-normalized, scaled, and shifted outputs.
 */
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

/**
 * @brief Applies a linear transformation to the input matrix.
 *
 * Computes Y = X * W + b, where:
 *   - X is an [m x in_dim] matrix of inputs,
 *   - W is an [in_dim x out_dim] weight matrix,
 *   - b is a vector of length out_dim representing the bias.
 *
 * @param X     Input matrix of shape [m x in_dim].
 * @param W     Weight matrix of shape [in_dim x out_dim].
 * @param b     Bias vector of length out_dim.
 * @return      Output matrix Y of shape [m x out_dim], where
 *              Y[i][j] = sum_{k=0..in_dim-1} X[i][k] * W[k][j] + b[j].
 */
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

/**
 * @brief Computes scaled dot-product attention with a causal mask.
 *
 * For each query in Q, this function computes attention scores against all keys in K:
 *   S = (Q ⋅ Kᵀ) / √d_k + mask
 * where mask contains large negative values to prevent attention to future positions.
 * It then applies softmax row-wise to S to obtain attention weights P, and finally
 * computes the output as P ⋅ V.
 *
 * @param Q     Query matrix of shape [n_q x d_k].
 * @param K     Key matrix of shape   [n_k x d_k].
 * @param V     Value matrix of shape [n_k x d_v].
 * @param mask  Causal mask matrix of shape [n_q x n_k], where mask[i][j] is
 *              typically 0 for allowed positions and a large negative number
 *              (e.g., -1e10) to suppress attention to disallowed (future) tokens.
 * @return      Attention output matrix of shape [n_q x d_v].
 */
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

    // Compute P * V => [n_q x d_v]
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

/**
 * @brief Performs multi-head self-attention with a causal mask.
 *
 * This function implements GPT-2's multi-head attention:
 * 1. Projects the input X into concatenated Q, K, V matrices.
 * 2. Splits the concatenated QKV into separate Q, K, V of shape [n_seq x n_embd].
 * 3. Constructs a causal mask that blocks attention to future tokens.
 * 4. Divides Q, K, V into n_head smaller heads and applies scaled dot-product
 *    attention independently for each head.
 * 5. Concatenates the per-head outputs back into a single [n_seq x n_embd] matrix.
 * 6. Applies a final linear projection to produce the output.
 *
 * @param X       Input matrix of shape [n_seq x n_embd].
 * @param Wqkv    Weight matrix for QKV projection, of shape [n_embd x (3*n_embd)].
 * @param bqkv    Bias vector for QKV projection, of length 3*n_embd.
 * @param Wproj   Weight matrix for the final output projection, of shape [n_embd x n_embd].
 * @param bproj   Bias vector for the final output projection, of length n_embd.
 * @param n_head  Number of attention heads (must divide n_embd evenly).
 * @return        Output matrix of shape [n_seq x n_embd], result of multi-head attention.
 */
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

    // 1) Project X => QKV combined: [n_seq x 3*n_embd]
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

    // 5) Concatenate all head outputs => [n_seq x n_embd]
    Matrix concat(n_seq, vector<float>(n_embd, 0.0f));
    for (int i = 0; i < n_seq; ++i) {
        for (int h = 0; h < n_head; ++h) {
            const int start = h * d_head;
            for (int j = 0; j < d_head; ++j) {
                concat[i][start + j] = head_outputs[h][i][j];
            }
        }
    }

    // 6) Final linear projection: [n_seq x n_embd] => [n_seq x n_embd]
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

/**
 * @brief Performs a single GPT-2 transformer block forward pass (pre-norm + residual).
 *
 * A transformer block consists of two sub-layers with pre-layer normalization:
 *  1) Multi-head self-attention:
 *     - Apply layer normalization to the input.
 *     - Compute multi-head attention (with causal masking).
 *     - Add the attention output back to the input (residual connection).
 *  2) Feed-forward network:
 *     - Apply layer normalization to the updated hidden states.
 *     - Apply a two-layer feed-forward network (Linear => GELU => Linear).
 *     - Add the feed-forward output back (residual connection).
 *
 * @param x_in    Input hidden states of shape [n_seq x n_embd].
 * @param W       BlockWeights struct containing:
 *                  - ln1_gamma, ln1_beta: LayerNorm1 parameters.
 *                  - Wqkv, bqkv: QKV projection weights and bias.
 *                  - Wproj, bproj: Attention output projection weights and bias.
 *                  - ln2_gamma, ln2_beta: LayerNorm2 parameters.
 *                  - W_fc, b_fc: Feed-forward first linear layer weights and bias.
 *                  - W_proj, b_proj: Feed-forward second linear layer weights and bias.
 * @param n_head  Number of attention heads (must evenly divide n_embd).
 * @return        Output hidden states after applying the transformer block,
 *                of shape [n_seq x n_embd].
 */
Matrix transformer_block(const Matrix& x_in,
                         const BlockWeights& W,
                         const int n_head)
{
    Matrix x = x_in;  // copy for residual

    // 1) LayerNorm => MHA => Residual
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

    // 2) LayerNorm => FFN => Residual
    Matrix norm2 = layer_norm(x, W.ln2_gamma, W.ln2_beta);
    // FFN: Linear => GELU => Linear
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

/**
 * @brief Performs the full forward pass of GPT-2 on a sequence of token IDs.
 *
 * This function implements GPT-2 inference:
 * 1. Embeds input tokens and adds positional embeddings.
 * 2. Applies a stack of transformer blocks (pre-norm + attention + feed-forward).
 * 3. Applies a final layer normalization.
 * 4. Projects hidden states to vocabulary logits using weight tying (Wteᵀ).
 *
 * @param input_tokens Vector of input token IDs of length n_seq.
 *                     Each ID must be in [0, model.n_vocab).
 * @param model        The loaded GPT2Model containing:
 *                         - Wte   : token embedding matrix [n_vocab x n_embd]
 *                         - Wpe   : positional embedding matrix [n_ctx x n_embd]
 *                         - blocks: vector of BlockWeights for each transformer layer
 *                         - ln_f_gamma, ln_f_beta: final layer norm parameters
 *                         - n_layer, n_head, n_embd, n_vocab
 * @return             A matrix of shape [n_seq x n_vocab], where each row
 *                     contains the logits for all vocabulary tokens at that position.
 */
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

    // 4) Output logits: x * Wte^T => [n_seq x n_vocab]
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

/**
 * @brief Reads and parses a JSON file into a nlohmann::json object.
 *
 * Opens the file at the specified path, parses its contents as JSON,
 * and returns the resulting json object.
 *
 * @param path  Path to the JSON file to read.
 * @return      A nlohmann::json object containing the parsed JSON data.
 *
 * @throws std::runtime_error if the file cannot be opened or if parsing fails.
 */
nlohmann::json read_json(const string& path) {
    ifstream fin(path);
    if (!fin || !fin.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + path);
    }
    nlohmann::json j;
    fin >> j;
    return j;
}

/**
 * @brief Reads raw float32 data from a binary file into a buffer.
 *
 * Opens the file at the given path in binary mode, resizes the output buffer
 * to expected_count, and reads exactly expected_count floats (4 bytes each).
 *
 * @param path            Path to the binary file containing float32 data.
 * @param[out] buffer     Vector<float> to be filled with the data. Will be resized
 *                        to expected_count.
 * @param expected_count  Number of float32 values expected to read from the file.
 *
 * @throws std::runtime_error if the file cannot be opened or if the read
 *                            operation fails to read the expected number of bytes.
 */
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

/**
 * @brief Loads a GPT-2 model's weights and configuration from disk.
 *
 * Expects the following files in <model_dir>:
 *   - hparams.json          (model hyperparameters: n_embd, n_layer, n_head, n_ctx, n_vocab)
 *   - metadata.json         (mapping of parameter names → shapes)
 *   - <param_name>.bin      (raw float32 weights for every entry in metadata.json)
 *   - encoder.json, vocab.bpe (tokenizer files; not loaded here)
 *
 * The function:
 *   1. Reads hparams.json to populate GPT2Model fields.
 *   2. Reads metadata.json to learn each parameter's shape.
 *   3. Defines helper lambdas to load 1D vectors and 2D matrices from .bin files.
 *   4. Loads token embeddings (Wte) and positional embeddings (Wpe).
 *   5. Loads final layer‐norm parameters (ln_f_gamma, ln_f_beta).
 *   6. Iterates over each transformer block and loads its LayerNorm,
 *      attention (QKV and output), and feed-forward weights/biases.
 *
 * @param model_dir Path to the directory containing the GPT-2 model files.
 * @return A fully populated GPT2Model struct with all weights and hyperparameters.
 * @throws std::runtime_error if any file is missing, malformed, or cannot be read.
 */
GPT2Model load_model(const string& model_dir) {
    // 1) Read hparams.json
    const string hparams_path = model_dir + "/hparams.json";
    const nlohmann::json hps = read_json(hparams_path);

    GPT2Model model;
    model.n_embd     = hps.at("n_embd").get<int>();
    model.n_layer    = hps.at("n_layer").get<int>();
    model.n_head     = hps.at("n_head").get<int>();
    model.n_ctx      = hps.at("n_ctx").get<int>();
    model.n_vocab    = hps.at("n_vocab").get<int>();

    // 2) Read metadata.json
    const string meta_path = model_dir + "/metadata.json";
    const nlohmann::json meta = read_json(meta_path);

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

        // Reshape flat => Matrix
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
        // Attention: c_attn => Wqkv, bqkv
        W.Wqkv      = load_matrix("blocks_" + std::to_string(layer) + "_attn_c_attn_w");
        W.bqkv      = load_vector("blocks_" + std::to_string(layer) + "_attn_c_attn_b");
        // Attention: c_proj => Wproj, bproj
        W.Wproj     = load_matrix("blocks_" + std::to_string(layer) + "_attn_c_proj_w");
        W.bproj     = load_vector("blocks_" + std::to_string(layer) + "_attn_c_proj_b");
        // LayerNorm 2
        W.ln2_gamma = load_vector("blocks_" + std::to_string(layer) + "_ln_2_g");
        W.ln2_beta  = load_vector("blocks_" + std::to_string(layer) + "_ln_2_b");
        // Feed-Forward: c_fc => W_fc, b_fc
        W.W_fc      = load_matrix("blocks_" + std::to_string(layer) + "_mlp_c_fc_w");
        W.b_fc      = load_vector("blocks_" + std::to_string(layer) + "_mlp_c_fc_b");
        // Feed-Forward: c_proj => W_proj, b_proj
        W.W_proj    = load_matrix("blocks_" + std::to_string(layer) + "_mlp_c_proj_w");
        W.b_proj    = load_vector("blocks_" + std::to_string(layer) + "_mlp_c_proj_b");
    }

    return model;
}

// ------------------------------------------------------------
//  Main Function: Load model, tokenize input, run inference
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

    // Tokenization step (BPE)
    BPEEncoder encoder(model_dir);
    vector<int> input_tokens = encoder.encode(prompt);

    // Run a single forward pass
    const Matrix logits = gpt2_forward(input_tokens, model);

    // Greedy decode: take the logits of the last position
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

    // Append the predicted ID and decode the full sequence back to text
    input_tokens.push_back(next_id);
    string output_text = encoder.decode(input_tokens);

    std::cout << "Decoded text:\n" << output_text << "\n";

    return 0;
}
