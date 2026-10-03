"""LoRA fine-tune NLLB-200-distilled-600M for English<->{yo,ha,ig} on MAFAND-MT.

Runs on a free Kaggle T4/P100 or Colab T4 (use fp16, batch 8, grad-accum 4).
    pip install -e ".[train,ml]"
    python training/finetune_nllb.py --lang yo --epochs 3 --out outputs/nllb-yo-lora

Dataset: masakhane/mafand (configs: en-yor, en-hau, en-ibo). Check the licence on the dataset
card before redistributing derived models.
"""
from __future__ import annotations

import argparse

import mlflow
import numpy as np
import sacrebleu
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

CODES = {"yo": ("yor_Latn", "en-yor"), "ha": ("hau_Latn", "en-hau"), "ig": ("ibo_Latn", "en-ibo")}
BASE = "facebook/nllb-200-distilled-600M"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=CODES, required=True)
    ap.add_argument("--direction", choices=["xx-en", "en-xx"], default="xx-en")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="outputs/nllb-lora")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)

    nllb_code, cfg = CODES[a.lang]
    src_l, tgt_l = ("eng_Latn", nllb_code) if a.direction == "en-xx" else (nllb_code, "eng_Latn")
    src_k, tgt_k = (("en", a.lang) if a.direction == "en-xx" else (a.lang, "en"))
    # MAFAND uses 'yo', 'ha', 'ig' / 'en' keys inside a 'translation' dict
    ds = load_dataset("masakhane/mafand", cfg)

    tok = AutoTokenizer.from_pretrained(BASE, src_lang=src_l, tgt_lang=tgt_l)

    def prep(batch):
        src = [t[src_k] for t in batch["translation"]]
        tgt = [t[tgt_k] for t in batch["translation"]]
        return tok(src, text_target=tgt, max_length=128, truncation=True)

    tokenized = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()

    def metrics(pred):
        preds, labels = pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        hyp = tok.batch_decode(preds, skip_special_tokens=True)
        ref = tok.batch_decode(labels, skip_special_tokens=True)
        return {"chrf++": sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score,
                "bleu": sacrebleu.corpus_bleu(hyp, [ref]).score}

    args = Seq2SeqTrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=8, per_device_eval_batch_size=16,
        gradient_accumulation_steps=4, fp16=True, eval_strategy="epoch", save_strategy="epoch",
        predict_with_generate=True, generation_max_length=128, logging_steps=50,
        load_best_model_at_end=True, metric_for_best_model="chrf++", save_total_limit=2,
        report_to=["mlflow"], seed=a.seed)

    mlflow.set_experiment("nllb-finetune")
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tokenized["train"],
                             eval_dataset=tokenized["validation"],
                             data_collator=DataCollatorForSeq2Seq(tok, model=model),
                             compute_metrics=metrics)
    trainer.train()
    print("TEST:", trainer.evaluate(tokenized["test"], metric_key_prefix="test"))
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
