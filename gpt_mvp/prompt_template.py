# Template for analyzing legal clauses with GPT-4o

CLAUSE_ANALYSIS_PROMPT = """
Analyze the following legal clause and provide a structured assessment.

CLAUSE TEXT:
"""""

{clause}
"""""

Provide your analysis in JSON format with the following fields:
1. "type": The type of clause (e.g., Indemnification, Confidentiality, Limitation of Liability, etc.)
2. "risk_score": A numerical score from 0.0 to 1.0 representing the level of legal risk (0.0 = low risk, 1.0 = high risk)
3. "explanation": A brief explanation of the risk assessment in plain English

Your response should follow this exact JSON structure:
```json
{
  "type": "[CLAUSE TYPE]",
  "risk_score": 0.X,
  "explanation": "[EXPLANATION]"
}
```

Basis for risk scoring:
- 0.0-0.3: Low risk - Standard, balanced clause with minimal legal exposure
- 0.4-0.6: Medium risk - Contains some concerning language but generally acceptable
- 0.7-0.9: High risk - Contains significant one-sided terms that favor the other party
- 1.0: Very high risk - Contains potentially unenforceable or highly problematic provisions

When determining the clause type and risk level, consider:
- Whether the clause is one-sided or balanced
- Industry standard practices for this type of clause
- Potential financial exposure or liability
- Clarity and specificity of obligations
- Presence of carve-outs or exceptions
"""
