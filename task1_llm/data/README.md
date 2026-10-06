# Task 1 data: TinyStories

The dataset is not committed. The team's shared [datasets.zip on Google Drive](https://drive.google.com/file/d/1Elz-CSlEQwb5e6X7R8mPWsrb1negSjpQ/view?usp=drive_link) (read access, see the root `README.md`) unpacks `TinyStories-train.txt` and `TinyStories-valid.txt` into this folder.

Alternatively, put one TinyStories file in this folder (run from the repository root):

```bash
wget -P task1_llm/data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
```

Parth's code (`task1_llm/parth/`): `.txt` (stories separated by `<|endoftext|>` or blank lines), `.jsonl`, and `.json` (records with a `text` or `story` field) are supported. With `data_file: null` in `task1_llm/parth/config/train.yaml`, the loader picks the first file in this folder whose name contains "train" (and not "valid"), so with the zip layout it uses `TinyStories-train.txt`. Parth's reported run used `TinyStoriesV2-GPT4-train.txt`; set `data_file` to choose a file explicitly.

In Parth's code the train/validation split is made at the story level with seed 8503 before the tokenizer is fit.
