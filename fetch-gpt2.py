#!/usr/bin/env python3
"""
fetch-gpt2.py

This script:
  1. Downloads (if needed) the GPT-2 model checkpoint and tokenizer files.
  2. Loads all model parameters from the TensorFlow checkpoint.
  3. Flattens the nested params dict into a mapping {flattened_name: ndarray}.
  4. Saves each ndarray as raw float32 binary (.bin).
  5. Writes a metadata.json that maps each flattened_name → shape.
  6. Copies the tokenizer files (encoder.json, vocab.bpe) to the output directory.

Usage:
    python fetch-gpt2.py

Optional Usage:
    python fetch-gpt2.py \
        --model_size 124M \
        --models_dir ./models \
"""

import os
import re
import json
import shutil
import argparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ------------------------------------------------------------------------------
# (1) Download GPT-2 files if they’re missing in models_dir/model_size
# ------------------------------------------------------------------------------

def download_gpt2_files(model_size, model_dir):
    """
    Download GPT-2 files (checkpoint + encoder.json + vocab.bpe) if missing.
    """
    assert model_size in ["124M", "355M", "774M", "1558M"]
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]
    os.makedirs(model_dir, exist_ok=True)

    for filename in filenames:
        out_path = os.path.join(model_dir, filename)
        if os.path.exists(out_path):
            continue  # already there

        url = f"{base_url}/{model_size}/{filename}"
        print(f"Downloading {filename} …")
        # use tf.keras.utils.get_file to stream/download
        local_path = tf.keras.utils.get_file(
            fname=filename,
            origin=url,
            cache_dir=model_dir,
            cache_subdir=".",
        )
        # tf may have placed it in ~/.keras/datasets; copy if needed
        if os.path.abspath(local_path) != os.path.abspath(out_path):
            shutil.copy(local_path, out_path)
        print(f"  → Saved to {out_path}")


# ------------------------------------------------------------------------------
# (2) Load TF checkpoint into a nested dict of NumPy arrays
# ------------------------------------------------------------------------------

def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    """
    Walk through all variables in the TF checkpoint at tf_ckpt_path,
    strip off the "model/" prefix, and insert them into a nested dict:
        params = {
            "wte": ndarray,
            "wpe": ndarray,
            "ln_f": { "g": ndarray, "b": ndarray },
            "blocks": [
                {
                    "ln_1": { "g": ndarray, "b": ndarray },
                    "ln_2": { "g": ndarray, "b": ndarray },
                    "attn": {
                        "c_attn": { "w": ndarray, "b": ndarray },
                        "c_proj": { "w": ndarray, "b": ndarray },
                    },
                    "mlp": {
                        "c_fc":   { "w": ndarray, "b": ndarray },
                        "c_proj": { "w": ndarray, "b": ndarray },
                    },
                },
                …  # for each layer
            ],
            // maybe other keys (if hparams has extras)
        }
    """
    def set_in_nested_dict(d, keys, val):
        """
        Recursively insert val at nested key path `keys` in dict d.
        If keys is empty, return val (to be assigned).
        """
        if len(keys) == 0:
            return val
        k = keys[0]
        if k not in d:
            d[k] = {}
        d[k] = set_in_nested_dict(d[k], keys[1:], val)
        return d

    # Create skeleton with "blocks" = list of empty dicts
    n_layer = int(hparams["n_layer"])
    params = {"blocks": [{} for _ in range(n_layer)]}

    # List all variables in the TF checkpoint
    var_list = tf.train.list_variables(tf_ckpt_path)
    for (name, shape) in var_list:
        # Load the variable as a NumPy array (squeeze out singleton dims)
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        if not name.startswith("model/"):
            continue
        short_name = name[len("model/") :]  # e.g. "h0/attn/c_attn/w"
        if short_name.startswith("h"):
            # Matches: "h{layer_ix}/{rest_of_path}"
            m = re.match(r"h([0-9]+)/(.*)", short_name)
            layer_ix = int(m.group(1))
            sub_name = m.group(2)  # e.g. "attn/c_attn/w"
            key_list = sub_name.split("/")  # ["attn","c_attn","w"]
            # Insert into params["blocks"][layer_ix]
            params["blocks"][layer_ix] = set_in_nested_dict(
                params["blocks"][layer_ix], key_list, array
            )
        else:
            # top-level key, e.g. "wte", "wpe", "ln_f/g", etc.
            key_list = short_name.split("/")
            params = set_in_nested_dict(params, key_list, array)

    return params


def load_encoder_hparams_and_params(model_size, models_dir):
    """
    Returns:
      - encoder: the GPT-2 BPE encoder (from encoder.py)
      - hparams: dict loaded from hparams.json
      - params: nested dict of all model parameters
    """
    assert model_size in ["124M", "355M", "774M", "1558M"]
    model_dir = os.path.join(models_dir, model_size)

    # (a) Download if needed
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    # (b) Load the BPE encoder
    from encoder import get_encoder
    encoder = get_encoder(model_size, models_dir)

    # (c) Load hparams.json
    with open(os.path.join(model_dir, "hparams.json"), "r") as f:
        hparams = json.load(f)

    # (d) Load all model parameters from the TF checkpoint
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)
    return encoder, hparams, params


# ------------------------------------------------------------------------------
# (3) Flatten nested dict so that each final entry is a single NumPy array
# ------------------------------------------------------------------------------

def flatten_params(d, parent_key="", sep="_", flat_dict=None):
    """
    Recursively flattens a nested structure of dicts and lists into flat_dict:
      flat_dict["parent_sep_child_sep_index_sep_subchild"] = ndarray
    """
    if flat_dict is None:
        flat_dict = {}

    # Case 1: d is a dict → recurse on its items
    if isinstance(d, dict):
        for key, val in d.items():
            new_key = parent_key + sep + key if parent_key else key
            flatten_params(val, new_key, sep=sep, flat_dict=flat_dict)

    # Case 2: d is a list → recurse on each element, using index as part of the key
    elif isinstance(d, list):
        for idx, item in enumerate(d):
            new_key = parent_key + sep + str(idx) if parent_key else str(idx)
            flatten_params(item, new_key, sep=sep, flat_dict=flat_dict)

    # Case 3: leaf (we expect a NumPy array or something convertible)
    else:
        flat_dict[parent_key] = d

    return flat_dict


# ------------------------------------------------------------------------------
# (4) Save each flattened array to <output_dir>/<name>.bin (raw float32). Also
#     produce metadata.json mapping name → shape.
# ------------------------------------------------------------------------------

def save_params_binary(flat_params, output_dir):
    """
    Given flat_params: { name (str): np.ndarray,  … }, write each as:
      output_dir/name.bin   (raw float32 in row-major)
    Then write output_dir/metadata.json = { name: [dim0,dim1,…], … }.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}

    for name, arr in flat_params.items():
        arr = np.array(arr, dtype=np.float32)
        shape = list(arr.shape)
        metadata[name] = shape

        bin_path = os.path.join(output_dir, f"{name}.bin")
        arr.tofile(bin_path)  # raw bytes, float32, row-major

    # Write metadata.json
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  → Saved {len(flat_params)} arrays as .bin, wrote metadata.json")

# ------------------------------------------------------------------------------
# (5) Main: glue everything together
# ------------------------------------------------------------------------------

def main(model_size, models_dir, output_dir):
    print("1) Loading encoder, hparams, and checkpoint parameters …")
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    print("2) Flattening parameters dict (num arrays = {}) …".format(
        sum(1 for _ in flatten_params(params).items()))
    )
    flat = flatten_params(params)

    print("3) Saving all parameters as raw float32 binaries …")
    save_params_binary(flat, output_dir)

    print("\nAll done! Your C++ code can now load:")
    print("  - <output_dir>/<model_size>/metadata.json   (maps name → shape)")
    print("  - <output_dir>/<model_size>/<name>.bin      (raw float32 data)")
    print("  - <output_dir>/<model_size>/encoder.json    (BPE encoder config)")
    print("  - <output_dir>/<model_size>/vocab.bpe       (BPE merges/vocab)")
    print("\nHappy inference in C++!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GPT-2 TF checkpoint → raw float32 binaries + metadata for C++."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="124M",
        choices=["124M", "355M", "774M", "1558M"],
        help="GPT-2 model size (124M, 355M, 774M, or 1558M).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="Directory where GPT-2 checkpoints (and encoder.json, vocab.bpe) are stored (or will be downloaded).",
    )
    args = parser.parse_args()
    output_dir = os.path.join(args.models_dir, args.model_size)
    main(args.model_size, args.models_dir, output_dir)
