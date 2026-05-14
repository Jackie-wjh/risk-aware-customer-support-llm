Dataset README
==============

Project
-------
Risk-Aware LLM for Customer Support

This dataset is designed for training and evaluating a customer-support decision system. 
The task is to classify each customer request into one of four action labels:

1. Answer
2. Ask Back
3. Escalate
4. Refuse

The goal is not only to generate fluent customer-service responses, but also to decide the safest and most appropriate support action in realistic business scenarios.


Files
-----

This folder contains the following files:

1. merged_all_updated_cleaned_split (1).xlsx
   - Full cleaned dataset with train / validation / test split information.
   - Includes generated columns such as normalized_instruction, group_id, risk_bucket, is_risk_stress_test, and split.
   - This file is mainly used for inspection, analysis, and reporting.

2. train.jsonl
   - Training set.
   - Used for model training, SFT, LoRA fine-tuning, classifier training, or verifier training.

3. val.jsonl
   - Validation set.
   - Used for prompt tuning, threshold tuning, checkpoint selection, and development evaluation.
   - This set can be used during model development.

4. test.jsonl
   - Main test set.
   - Used only for final evaluation.
   - This file should not be used for prompt tuning, threshold tuning, or model selection.

5. test_high_risk_only.jsonl
   - A high-risk-only evaluation subset derived from test.jsonl.
   - It contains only sensitive business-domain cases:
     payment/refund, account security, complaint, human-agent request, and refusal cases.
   - It excludes normal and missing_info-only cases.
   - No labels or sample contents were modified when creating this subset.


Dataset Size
------------

Total samples: 27,030

Final split:

Train:      18,917
Validation:  4,056
Test:        4,057


Label Distribution
------------------

Train set:

Answer:    11,243
Ask Back:   6,376
Escalate:   1,263
Refuse:        35

Validation set:

Answer:     2,410
Ask Back:   1,363
Escalate:     273
Refuse:        10

Test set:

Answer:     2,410
Ask Back:   1,363
Escalate:     272
Refuse:        12


High-Risk-Only Test Set
-----------------------

The high-risk-only test set is a filtered subset of test.jsonl.

Total samples: 1,402

Risk bucket distribution:

payment_refund:    748
account_security:  301
human_agent:       188
complaint:         153
refuse:             12

Label distribution:

Answer:    791
Ask Back:  341
Escalate:  258
Refuse:     12

Important:
The high-risk-only test set is not a modified dataset.
It is only a filtered subset of the original test set.
The instruction, response, label, category, and intent fields are unchanged.


Label Definitions
-----------------

Answer:
Use this label when the customer request is legitimate, low-risk, and can be handled with a direct answer, standard policy, or normal support process.

Ask Back:
Use this label when the customer request is legitimate but lacks required information. 
The safest next step is to ask the customer for missing details before continuing.

Escalate:
Use this label when the customer request is legitimate but sensitive, high-risk, case-specific, or requires human support, manual review, specialist handling, secure verification, or investigation.

Refuse:
Use this label when the request is unsafe, unauthorized, privacy-violating, fraudulent, or asks to bypass security or policy restrictions.


Split Strategy
--------------

The dataset was split using a group-based and risk-aware strategy rather than a simple random split.

The group_id was created from:

category + intent + normalized_instruction

This prevents identical or near-duplicate instruction templates from appearing across train, validation, and test sets.

The final split follows an approximately 70 / 15 / 15 ratio:

Train:      70%
Validation: 15%
Test:       15%

Since Refuse is a rare but safety-critical label, it was manually preserved across splits:

Train Refuse:      35
Validation Refuse: 10
Test Refuse:       12


Data Quality Checks
-------------------

The final dataset passed the following checks:

1. Total row count remains 27,030.
2. All labels are valid:
   - Answer
   - Ask Back
   - Escalate
   - Refuse
3. No duplicate label conflicts remain.
4. No group leakage exists across train, validation, and test.
5. Answer, Ask Back, Escalate, and Refuse distributions are stable across splits.
6. Refuse samples are preserved in validation and test.
7. The high-risk-only test set is derived only from the test set.


JSONL Format
------------

Each JSONL file contains one JSON object per line.

Example format:

{
  "instruction": "...",
  "category": "...",
  "intent": "...",
  "response": "...",
  "label": "Ask Back",
  "risk_bucket": "missing_info",
  "is_risk_stress_test": 1
}


Recommended Usage
-----------------

Use train.jsonl for training.

Use val.jsonl for:
- prompt tuning
- threshold tuning
- checkpoint selection
- validation evaluation

Use test.jsonl only for final main evaluation.

Use test_high_risk_only.jsonl for final high-risk-domain evaluation.

Do not use test.jsonl or test_high_risk_only.jsonl during model development or tuning.


Evaluation Recommendation
-------------------------

For the main test set, report:

- Accuracy
- Macro-F1
- Per-class precision, recall, and F1
- Confusion matrix

For the high-risk-only test set, additionally focus on:

- Escalate recall
- Refuse recall
- Unsafe Answer rate
- High-risk error rate
- Cases where Escalate or Refuse is incorrectly predicted as Answer


Notes
-----

Answer and Ask Back are not forced to be equally balanced.
The goal is to preserve realistic business distribution while keeping label distributions stable across train, validation, and test.

The test set should remain frozen after this split.
Any further prompt, threshold, or model selection should be done only on the validation set.