import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
import os

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# カスタムデータセットクラス
class CommentDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_text = self.examples[i]['input']
        output_text = self.process_comment(self.examples[i]['output'])
        combined_text = f"字幕: {input_text}\nコメント: {output_text}"
        encodings = self.tokenizer(combined_text, truncation=True, max_length=self.block_size, padding="max_length")
        return {"input_ids": torch.tensor(encodings["input_ids"], device=device),
                "attention_mask": torch.tensor(encodings["attention_mask"], device=device)}

    def process_comment(self, comment):
        comment = comment.split('\uEE06')[0]
        return comment[:20]

# チェックポイントを取得する関数
def get_latest_checkpoint(output_dir):
    checkpoints = [dir for dir in os.listdir(output_dir) if dir.startswith('checkpoint-')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(output_dir, latest_checkpoint)

# モデルの学習関数
def train_model(model_name, train_file, eval_file, output_dir, num_train_epochs=3, save_steps=1000):
    # モデルとトークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # データセットの準備
    train_dataset = CommentDataset(tokenizer, train_file, block_size=256)
    eval_dataset = CommentDataset(tokenizer, eval_file, block_size=256)

    # トレーニングの設定
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        save_steps=save_steps,  # チェックポイントの保存頻度
        save_total_limit=3,  # 保存するチェックポイントの最大数
        fp16=True  # GPU使用時の16ビット精度学習を有効化
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 最新のチェックポイントを探す
    latest_checkpoint = get_latest_checkpoint(output_dir)

    # チェックポイントがある場合は読み込む
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.train()

    # モデルの保存
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

# コメント生成関数
def generate_comments(model, tokenizer, subtitle, num_comments=5):
    input_text = f"字幕: {subtitle}\nコメント:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 30,
        num_return_sequences=num_comments,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    comments = []
    for sequence in output:
        decoded = tokenizer.decode(sequence, skip_special_tokens=True)
        comment = decoded.split("コメント:")[-1].strip()
        comment = process_comment(comment)
        comments.append(comment)
    return comments

def process_comment(comment):
    comment = comment.split('\uEE06')[0]
    return comment[:20]

if __name__ == "__main__":
    model_name = "rinna/japanese-gpt2-xsmall"
    train_file = "train_data.jsonl"
    eval_file = "eval_data.jsonl"
    output_dir = "./drive/MyDrive/content/niconico_comment_model"

    train_model(model_name, train_file, eval_file, output_dir, num_train_epochs=3, save_steps=1000)

    # 学習済みモデルの読み込み
    loaded_model = AutoModelForCausalLM.from_pretrained(f"{output_dir}/final_model").to(device)
    loaded_tokenizer = AutoTokenizer.from_pretrained(f"{output_dir}/final_model")

    # 使用例
    subtitle = "主人公が最終決戦に挑む"
    generated_comments = generate_comments(loaded_model, loaded_tokenizer, subtitle)
    print(f"字幕: {subtitle}")
    print("生成されたコメント:")
    for comment in generated_comments:
        print(comment)