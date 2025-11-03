'''
Author : yj z
Created: 2025-11-02 20:02
File   : pretrainDataset.py
Description: pretrain 数据集读取，读取text字段
Version: 1.0.0
'''




import json
import torch
from torch.utils.data import Dataset,DataLoader

class PretrainDataset(Dataset):
    """
    用于自回归语言模型预训练的Dataset：
    - 输入为JSONL文件，每行包含字段 "text"
    - 输出 (X, Y, loss_mask)
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    def _load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "text" in data:
                        samples.append(data["text"])
                    else:
                        print(f"[WARN] 第 {line_num} 行没有 'text' 字段")
                except json.JSONDecodeError as e:
                    print(f"[WARN] 第 {line_num} 行解析错误: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = str(self.samples[index])

        # ====== 编码 ======
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ====== 自回归shift处理 ======
        X = input_ids[:-1]
        Y = input_ids[1:]
        loss_mask = attention_mask[1:]

        return {
            "input_ids": X.to(torch.long),
            "labels": Y.to(torch.long),
            "loss_mask": loss_mask.to(torch.long),
        }
                   
def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../VerMind")

    dataset = PretrainDataset(
        data_path="./test_pretrain.jsonl",
        tokenizer=tokenizer,
        max_length=20
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))

    print("\n--- Dataset 输出检查 ---")
    print("input_ids:", batch["input_ids"].shape)
    print("labels:", batch["labels"].shape)
    print("loss_mask:", batch["loss_mask"])

    print("\n--- 解码检查 ---")
    print("Input :", tokenizer.decode(batch["input_ids"][0]))
    print("Target:", tokenizer.decode(batch["labels"][0][batch["labels"][0] != -100]))

if __name__ == "__main__":
    main()