import json

path = 'c:/Users/Asus/Downloads/Ecommerce-LLM-Finetuning/Ecommerce-LLM-Finetuning/notebooks/03_preprocessing.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix duplicate lines in the first cell and import Preprocessor in cell 5
nb['cells'][5]['source'] = [
    "from src.preprocess import Preprocessor\n",
    "preprocessor = Preprocessor(min_words=2, max_words=400)\n",
    "df_clean = preprocessor.run(df_raw, lowercase=False)\n",
    "\n",
    "print(f\"\\nFinal clean dataset: {len(df_clean):,} rows (from {len(df_raw):,} raw rows)\")\n",
    "df_clean.head(5)\n"
]

nb['cells'][9]['source'] = [
    "from src.utils import save_jsonl\n",
    "out_path = cfg.processed_data_dir / \"cleaned_pairs.jsonl\"\n",
    "save_jsonl(df_clean.to_dict(orient=\"records\"), out_path)\n",
    "\n",
    "# Also keep a small human-readable CSV sample for quick spot-checks\n",
    "sample_path = cfg.sample_data_dir / \"cleaned_sample.csv\"\n",
    "df_clean.sample(min(50, len(df_clean)), random_state=cfg.seed).to_csv(sample_path, index=False)\n",
    "print(f\"Saved sample preview -> {sample_path}\")\n"
]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook fixed.")
