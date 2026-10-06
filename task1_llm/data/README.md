# Task 1 data: TinyStories

The dataset is not committed. Put one TinyStories file in this folder (run from the repository root):

```bash
wget -P task1_llm/data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
```

`.txt` (stories separated by `<|endoftext|>` or blank lines), `.jsonl`, and `.json` (records with a `text` or `story` field) are supported. With `data_file: null` in `task1_llm/parth/config/train.yaml`, the loader picks the first file whose name contains "train".

The train/validation split is made at the story level with seed 8503 before the tokenizer is fit.
