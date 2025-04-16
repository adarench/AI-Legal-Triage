import openai
import json
from typing import Dict, Any, List
from .prompt_template import CLAUSE_ANALYSIS_PROMPT

class GPTClauseAnalyzer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        
    def analyze_clause(self, clause_text: str) -> Dict[str, Any]:
        """Analyze a legal clause using GPT-4o and return structured results."""
        # Prepare prompt with the specific clause
        prompt = CLAUSE_ANALYSIS_PROMPT.format(clause=clause_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,  # Lower temperature for more consistent outputs
                messages=[
                    {"role": "system", "content": "You are a legal expert who analyzes contract clauses."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            
            # Validate response structure
            required_keys = ["type", "risk_score", "explanation"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Response missing required key: {key}")
                    
            # Ensure risk_score is a float between 0 and 1
            risk_score = float(result["risk_score"])
            if not 0 <= risk_score <= 1:
                raise ValueError(f"Risk score must be between 0 and 1, got {risk_score}")
                
            result["risk_score"] = risk_score
            
            return result
            
        except Exception as e:
            return {
                "type": "Error",
                "risk_score": 0.0,
                "explanation": f"Failed to analyze clause: {str(e)}"
            }
            
    def batch_analyze(self, clauses: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple clauses in batch."""
        return [self.analyze_clause(clause) for clause in clauses]
