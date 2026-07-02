# 🛍️ Ecommerce-LLM-Finetuning

An end-to-end, **free-resources-only** pipeline for fine-tuning an open-source
LLM into an Ecommerce Customer Support Assistant — from raw data to a
deployed Streamlit chatbot.

> Built to run entirely on **Google Colab Free (T4 GPU)** using QLoRA
> (4-bit quantized LoRA fine-tuning) via **Unsloth**, **PEFT**, and **TRL**.

---

## 📌 Project Overview

| | |
|---|---|
| **Task** | Fine-tune an instruct LLM to answer ecommerce customer support questions (orders, shipping, refunds, returns, payments, coupons, delivery, account management) |
| **Base model** | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (swappable — see `src/config.py`) |
| **Method** | QLoRA (4-bit quantization + LoRA adapters) via Unsloth + PEFT + TRL's `SFTTrainer` |
| **Dataset** | Free Hugging Face ecommerce/customer-support dataset (with a synthetic fallback if offline) |
| **Compute** | Google Colab Free tier (T4 GPU, ~15GB VRAM) |
| **Serving** | Streamlit chat app, runs locally or on Streamlit Community Cloud |

---

## 🖼️ Screenshots

> _Add screenshots after running the app locally:_

```
docs/screenshots/chat_ui.png
docs/screenshots/training_loss.png
docs/screenshots/eval_comparison.png
```

---

## 🏗️ Architecture

```
                ┌────────────────────┐
                │  Hugging Face Hub   │
                │ (ecommerce dataset) │
                └─────────┬───────────┘
                          │  01_download_dataset.ipynb
                          ▼
                ┌────────────────────┐
                │   data/raw/         │
                └─────────┬───────────┘
                          │  02_data_exploration.ipynb
                          │  03_preprocessing.ipynb
                          ▼
                ┌────────────────────┐
                │ data/processed/     │  (clean instruction/response pairs)
                └─────────┬───────────┘
                          │  04_prompt_formatting.ipynb
                          ▼
                ┌────────────────────┐
                │ train/val/test      │  (formatted + tokenized)
                │ JSONL splits        │
                └─────────┬───────────┘
                          │  05_finetune_llm.ipynb  (QLoRA via Unsloth)
                          ▼
                ┌────────────────────┐
                │ models/lora_adapter │
                └─────────┬───────────┘
                 ┌────────┴─────────┐
      06_evaluation.ipynb   07_inference.ipynb (merge)
                 │                  │
                 ▼                  ▼
     BLEU / ROUGE / PPL     models/merged_model/
                                    │
                                    ▼
                        ┌────────────────────┐
                        │ app/streamlit_app.py│
                        │  (chat UI)           │
                        └────────────────────┘
```

**Code layout principle:** every notebook is a thin orchestration layer
over reusable, testable modules in `src/` (`data_loader.py`,
`preprocess.py`, `prompt_template.py`, `trainer.py`, `inference.py`,
`evaluate_model.py`, `utils.py`, `config.py`). The Streamlit app in `app/`
reuses the exact same `src/inference.py` and `src/prompt_template.py`
modules, so training and serving never drift apart.

---

## 📂 Folder Structure

```
Ecommerce-LLM-Finetuning/
│
├── data/
│   ├── raw/                # downloaded raw dataset
│   ├── processed/          # cleaned pairs + train/val/test splits + eval outputs
│   └── sample/              # small CSV/JSONL previews for quick inspection
│
├── notebooks/
│   ├── 01_download_dataset.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_prompt_formatting.ipynb
│   ├── 05_finetune_llm.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_inference.ipynb
│
├── models/
│   ├── base_model/          # (optional local cache of the base model)
│   ├── lora_adapter/        # saved LoRA adapter + tokenizer (small, ~50-200MB)
│   └── merged_model/        # base + LoRA merged, standalone model for serving
│
├── src/
│   ├── config.py            # central Config dataclass (paths, hyperparams)
│   ├── data_loader.py       # HF download + synthetic fallback + splitting
│   ├── preprocess.py        # cleaning, dedup, instruction-response pairing
│   ├── prompt_template.py   # prompt formatting (training + inference, shared)
│   ├── trainer.py           # Unsloth model load + LoRA + SFTTrainer wrapper
│   ├── inference.py         # load model (adapter or merged) + generate()
│   ├── evaluate_model.py    # BLEU / ROUGE / perplexity / comparison table
│   └── utils.py             # logging, GPU detection, text-cleaning helpers
│
├── app/
│   ├── streamlit_app.py     # main chat UI
│   ├── chatbot.py           # chat orchestration + topic guardrail + PDF FAQ
│   ├── model_loader.py       # cached model loading for Streamlit
│   └── assets/
│
├── requirements.txt
├── environment.yml
├── README.md
├── .gitignore
└── LICENSE
```

---

## ⚙️ Installation

### Option A — pip (recommended for the Streamlit app / local CPU work)

```bash
git clone <your-fork-url> Ecommerce-LLM-Finetuning
cd Ecommerce-LLM-Finetuning
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ `bitsandbytes` and `unsloth` require an NVIDIA CUDA GPU for training.
> The Streamlit **app** can run on CPU if you point it at a merged model
> (generation will just be slower).

### Option B — conda

```bash
conda env create -f environment.yml
conda activate ecommerce-llm-finetuning
```

---

## ☁️ Google Colab Instructions (Training)

1. Upload the whole repo to Google Drive, or `git clone` it inside Colab:
   ```python
   !git clone <your-fork-url>
   %cd Ecommerce-LLM-Finetuning
   ```
2. **Runtime → Change runtime type → Hardware accelerator → T4 GPU**
3. Run the notebooks **in order**, top to bottom:
   1. `01_download_dataset.ipynb`
   2. `02_data_exploration.ipynb`
   3. `03_preprocessing.ipynb`
   4. `04_prompt_formatting.ipynb`
   5. `05_finetune_llm.ipynb` ← the main fine-tuning step (QLoRA, ~20-45 min on T4 for a few epochs over a few thousand examples)
   6. `06_evaluation.ipynb`
   7. `07_inference.ipynb` ← merges the adapter into `models/merged_model/`
4. Download `models/merged_model/` (or `models/lora_adapter/`) locally, or
   push it to the Hugging Face Hub (see notebook 07, step 6) to load it
   from the Streamlit app without re-uploading large files manually.

**Colab Free tips:**
- Sessions disconnect after inactivity — keep the tab active during training.
- If you hit an OOM error, lower `per_device_train_batch_size` in
  `src/config.py` (e.g. to 1) and raise `gradient_accumulation_steps`.
- 4-bit QLoRA on a 1B-3B model comfortably fits in the T4's ~15GB VRAM.

---

## 🖥️ Streamlit Instructions (Serving)

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

**Features:**
- Chat interface with word-by-word typing animation
- Full conversation history in-session
- "Clear chat" button
- Temperature & max-tokens sliders
- Optional FAQ PDF upload for grounded, store-specific answers
- Responsive, clean UI with a scoped custom theme

The app auto-loads (in priority order): a merged model in
`models/merged_model/` → a LoRA adapter in `models/lora_adapter/` → the
untuned base model (with a warning), so it always boots even before you've
trained anything — useful for UI development.

---

## 🚀 Inference (Programmatic)

```python
from src.config import Config
from src.inference import EcommerceSupportBot

cfg = Config()
bot = EcommerceSupportBot(cfg)
bot.load_merged_model()   # or bot.load_adapter_model()

answer = bot.generate("Where is my order #48213?", temperature=0.6)
print(answer)
```

---

## 🏋️ Training (Programmatic)

```python
from datasets import Dataset
from src.config import Config
from src.trainer import EcommerceFineTuner
from src.utils import load_jsonl

cfg = Config()
train_ds = Dataset.from_list(load_jsonl(cfg.processed_data_dir / "train.jsonl"))
val_ds = Dataset.from_list(load_jsonl(cfg.processed_data_dir / "val.jsonl"))

ft = EcommerceFineTuner(cfg)
ft.load_base_model()
ft.apply_lora()
ft.build_trainer(train_ds, val_ds)
ft.train()
ft.save_adapter()
```

---

## 📊 Evaluation Metrics

Notebook `06_evaluation.ipynb` reports, on a held-out test sample:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — n-gram / longest-common-subsequence overlap
- **BLEU** (via sacreBLEU) — corpus-level precision-based overlap
- **Perplexity** — how well the model predicts held-out text
- **Ground Truth vs. Prediction table** — saved to
  `data/processed/eval_comparison_table.csv` for qualitative review

Fill in your own numbers after training, e.g.:

| Metric | Score |
|---|---|
| ROUGE-1 | _fill in_ |
| ROUGE-L | _fill in_ |
| BLEU | _fill in_ |
| Perplexity | _fill in_ |

---

## 🔧 Configuration

All paths and hyperparameters live in one place: `src/config.py`
(`Config` dataclass). Common things to tweak:

- `base_model_name` — swap between Llama-3.2-1B/3B, Qwen2.5-3B, Gemma-2B, TinyLlama
- `lora_r`, `lora_alpha`, `lora_target_modules` — LoRA capacity
- `num_train_epochs`, `learning_rate`, `per_device_train_batch_size`,
  `gradient_accumulation_steps` — training schedule
- `hf_dataset_name` — swap the source Hugging Face dataset

---

## 🧠 Chatbot Personality & Scope

The assistant is instructed (via system prompt + a lightweight keyword
guardrail in `app/chatbot.py`) to answer only ecommerce support topics:
order tracking, shipping, refunds, returns, payment issues, coupons,
delivery, and account management. Off-topic questions get a polite
redirect rather than a hallucinated answer.

---

## 🔮 Future Improvements

- Add retrieval-augmented generation (RAG) over a real store knowledge base
  instead of the single-PDF FAQ upload
- Multi-turn conversation memory in the model context (not just UI history)
- Quantized GGUF export for CPU-only / llama.cpp serving
- Human preference fine-tuning (DPO) on top of the SFT adapter
- Automated CI eval regression checks on every adapter update
- Multi-lingual support

---

## 📜 License

Project code is MIT licensed — see `LICENSE`. Base model weights and the
training dataset remain subject to their own upstream licenses (check the
specific Hugging Face model/dataset cards you choose in `src/config.py`).
