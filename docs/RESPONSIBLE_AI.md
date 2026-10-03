# Responsible AI statement

**Intended use:** general maternal health *information* for education. Not diagnosis, triage, or treatment.

**Safeguards**
1. Deterministic red-flag layer runs before any LLM call; CI fails if danger-sign recall < 95%.
2. Grounded answers only: retrieval score threshold; otherwise a fixed "ask a health worker" reply.
3. No drug names or doses; fixed disclaimer on every answer.
4. Privacy: synthetic data only in development; no raw user text in logs; no audio retention.
5. Translation risk: machine translation of medical text can be wrong. Escalation and refusal texts
   must be reviewed by native-speaker health workers before any real deployment.

**Known limitations** (update with measured numbers): low-resource-language STT/MT error rates,
dialect and accent variation, code-switching with Pidgin/English, no Pidgin support in NLLB,
possible outdated guidelines, LLM may still paraphrase unfaithfully.

**Before real-world use:** clinical review, native-speaker evaluation, ethics approval, data-protection
(NDPA 2023) assessment, pilot with human oversight.
