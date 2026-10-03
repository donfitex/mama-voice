---
language: [yo, ha, ig, en]
license: cc-by-nc-4.0   # NLLB is CC-BY-NC; check dataset licences too
base_model: facebook/nllb-200-distilled-600M
library_name: peft
tags: [translation, african-languages, lora, maternal-health]
---
# NLLB-600M LoRA: English <-> {Yoruba, Hausa, Igbo}

**Training data:** MAFAND-MT (state splits and licence).  **Method:** LoRA r=16, lr 2e-4, 3 epochs, T4.
**Evaluation:** FLORES-200 devtest, chrF++ / spBLEU (see table).

| Direction | Base chrF++ | Fine-tuned chrF++ | Base spBLEU | Fine-tuned spBLEU |
|---|---|---|---|---|
| yo -> en | TODO | TODO | TODO | TODO |

**Limitations:** domain shift to health text; news-heavy training data; not validated for clinical use.
Add a small in-domain maternal-health test set reviewed by native speakers.
