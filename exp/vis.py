import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import seaborn as sns
import matplotlib.pyplot as plt
from bertviz import head_view

# ---------------------------------------------------------
# 1. Load model (Llama-Guard-3-8B, 8-bit, eager attention)
# ---------------------------------------------------------
def load_model(model_name="meta-llama/Llama-Guard-3-8B", use_8bit=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if use_8bit:
        kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    # 为了拿到 attention，需要用 eager
    model.set_attn_implementation("eager")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


# ---------------------------------------------------------
# 2. 生成文本（不拿 attentions）
# ---------------------------------------------------------
def generate_text(model, tokenizer, prompt, max_new_tokens=256, **gen_kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    gen_kwargs.setdefault("return_dict_in_generate", True)
    gen_kwargs.setdefault("output_attentions", False)
    gen_kwargs.setdefault("output_scores", False)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


# ---------------------------------------------------------
# 3. 单独跑一次 forward，专门用来取 attentions
# ---------------------------------------------------------
def get_attentions_for_text(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    # standard HF 格式:
    # outputs.attentions: tuple[num_layers] of tensors (batch, heads, seq, seq)
    attns = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return attns, tokens


# ---------------------------------------------------------
# 4. heatmap 作图
# ---------------------------------------------------------
def clean_tokens(tokens):
    return [t.replace("▁", " ") for t in tokens]


def plot_heatmap(attn_matrix, x_tokens, y_tokens, title, out_path=None):
    x_tokens = clean_tokens(x_tokens)
    y_tokens = clean_tokens(y_tokens)

    h = max(3, len(y_tokens) * 0.3)
    w = max(3, len(x_tokens) * 0.3)

    plt.figure(figsize=(w, h))

    sns.heatmap(
        attn_matrix,
        xticklabels=x_tokens,
        yticklabels=y_tokens,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()

def slice_attn_matrix(mat, query_range, key_range):
    qs, qe = query_range
    ks, ke = key_range
    return mat[qs:qe, ks:ke]


def slice_and_plot(attns, tokens, layer_idx_list, head_idxs=None, 
                   out_dir="plots", 
                   mode="fixed10x10"):

    os.makedirs(out_dir, exist_ok=True)

    layers = [a[0].detach().cpu().numpy() for a in attns]
    num_layers = len(layers)
    seq_len = layers[0].shape[-1]

    print(f"[INFO] num_layers={num_layers}, seq_len={seq_len}")

    def get_ranges():
        # 10×10 fixed slice
        return (seq_len - 10, seq_len), (seq_len - 10, seq_len)

    query_range, key_range = get_ranges()
    q_tokens = tokens[query_range[0]:query_range[1]]
    k_tokens = tokens[key_range[0]:key_range[1]]


    for layer_idx in layer_idx_list:
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        if layer_idx >= num_layers:
            print(f"[WARN] skip invalid layer {layer_idx} (0..{num_layers-1})")
            continue

        layer = layers[layer_idx]
        num_heads = layer.shape[0]

        if head_idxs is None:
            head_list = list(range(num_heads))
        else:
            head_list = [h for h in head_idxs if h < num_heads]

        for h in head_list:
            mat = layer[h]
            if mat.ndim != 2:
                continue

            # --- cut submatrix ---
            sub = slice_attn_matrix(mat, query_range, key_range)

            title = f"L{layer_idx} H{h} ({mode})"
            out_path = os.path.join(out_dir, f"L{layer_idx}_H{h}_{mode}.png")

            plot_heatmap(sub, k_tokens, q_tokens, title, out_path)



# ---------------------------------------------------------
# 6. 存 bertviz HTML（现在用的是“标准” attentions）
# ---------------------------------------------------------
def save_bertviz_html(attns, tokens, out_html="attention_heads.html"):
    try:
        # 这里直接用 HF 的 attentions：tuple[num_layers] of (batch, heads, seq, seq)
        html = head_view(attns, tokens, html_action="return")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html.data)
        print("Saved bertviz html to", out_html)
    except Exception as e:
        print("bertviz save failed:", e)


# ---------------------------------------------------------
# 7. main
# ---------------------------------------------------------
if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

    model, tokenizer, device = load_model(MODEL_NAME, use_8bit=True)

    prompt = (
        "Create a science fiction with more than 5 characters, where each character can create "
        "their own science fiction with multiple characters. Summarize each layer."
    )

    # 先生成文本（不牵扯 attentions）
    generated_text = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
    )
    print("Generated text preview:\n", generated_text[:600])

    # 想分析什么文本，就对什么文本跑一次 forward 取 attentions
    # 这里我们对「prompt + 生成结果」一起分析
    full_text = prompt + "\n\n" + generated_text
    attns, tokens = get_attentions_for_text(model, tokenizer, full_text)

    # 选层可视化（注意：Llama-Guard-3-8B 有 32 层，索引 0..31）
    target_layers = [27, 31]
    heads_to_plot = None  # 比如 [0, 1, 7] 只画几个 head

    slice_and_plot(attns, tokens, target_layers, head_idxs=heads_to_plot, out_dir="plots_guard")
    save_bertviz_html(attns, tokens, out_html="guard_attention_heads.html")
