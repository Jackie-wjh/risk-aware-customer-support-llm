# Risk-Aware LLM for Customer Support

This repository contains the code, evaluation scripts, and final outputs for a risk-aware customer-support LLM project. The task is to build a small domain-adapted model that can decide how a customer query should be handled and then generate a response consistent with that decision.

The system predicts one of four operational actions:

- **Answer**: provide a direct response when the request is safe and sufficiently clear.
- **Ask Back**: ask for missing information when the request is legitimate but incomplete.
- **Escalate**: route the case to human or specialist support when the issue is high-risk or requires manual handling.
- **Refuse**: decline unsafe, unauthorised, or policy-violating requests.

## Final Model Route

The final post-training route uses **Qwen2.5-3B-Instruct** with supervised fine-tuning and **LoRA/PEFT**. The model is trained to produce a structured action-and-response output, including:

1. an action label,
2. a short decision reason,
3. a customer-facing response.

This design allows the model to learn both the decision layer and the response layer jointly. It is especially focused on difficult risk-aware boundaries such as **Ask Back vs Escalate** and **Escalate vs Refuse**.

## Evaluation

The project evaluates both decision quality and response quality:

- **Eval-A: Action Decision Evaluation**  
  Measures action-level performance using accuracy, macro-F1, per-class recall, confusion matrices, unsafe answer rate, and high-risk error patterns.

- **Eval-B: Response Quality Evaluation**  
  Uses an LLM judge to score generated responses across response quality dimensions such as action consistency, helpfulness, safety, professionalism, and hallucination control.

- **Composite Scoring**  
  Combines decision correctness and response quality into a final score for model comparison and selection.

- **High-Risk Stress Test**  
  Further evaluates the strongest models on sensitive cases involving payment/refund, account security, complaints, human-agent requests, and refusal-related scenarios.

## Main Outputs

This repository includes:

- training and inference scripts,
- Eval-A decision-level results,
- Eval-B LLM-judge results,
- final composite score outputs,
- high-risk stress-test results,
- final comparison summaries and reports.