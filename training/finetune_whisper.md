# Whisper fine-tuning (optional stretch)

Free route: Kaggle/Colab T4, `openai/whisper-small` + LoRA (peft), Common Voice (yo, ha) and
Google FLEURS (yo_ng, ha_ng). Follow the HF "Fine-Tune Whisper" blog recipe, swap in
`LoraConfig(r=32, target_modules=["q_proj","v_proj"])`, train with `Seq2SeqTrainer`,
and report WER/CER with `eval/stt_eval.py` on the FLEURS test split BEFORE and AFTER.
Igbo: Whisper has no Igbo support, so keep MMS adapters or fine-tune Whisper from a related checkpoint.
