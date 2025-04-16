# Maps clause types to risk scores based on legal domain knowledge
# These scores are estimates and should be adjusted by legal experts

RISK_MAP = {
    "Affiliate License-Licensee": 0.4,
    "Affiliate License-Licensor": 0.3,
    "Anti-Assignment": 0.6,
    "Audit Rights": 0.5,
    "Auto Renewal": 0.7,
    "Change of Control": 0.8,
    "Competitive Restriction Exception": 0.4,
    "Competitive Restrictions": 0.8,
    "Confidentiality Exception-Compelled Disclosure": 0.5,
    "Confidentiality Exceptions": 0.5,
    "Confidentiality Term": 0.4,
    "Confidentiality": 0.6,
    "Damage Cap": 0.9,
    "Effective Date": 0.1,
    "Exclusivity": 0.8,
    "Expiration Date": 0.2,
    "Governing Law": 0.5,
    "Indemnification": 0.9,
    "Insurance": 0.6,
    "IP Ownership Assignment": 0.8,
    "Limitation of Liability": 0.9,
    "Limited License": 0.6,
    "Liquidated Damages": 0.8,
    "Minimum Commitment": 0.7,
    "Minimum Term": 0.5,
    "Most Favored Nation": 0.7,
    "No-Solicit Of Customers": 0.8,
    "No-Solicit Of Employees": 0.7,
    "Non-Compete": 0.8,
    "Non-Disparagement": 0.6,
    "Post-Term Services": 0.5,
    "Price Restrictions": 0.7,
    "Product Warranty": 0.6,
    "Renewal Term": 0.5,
    "Revenue/Profit Sharing": 0.7,
    "Source Code Escrow": 0.6,
    "Termination For Convenience": 0.8,
    "Termination For Insolvency": 0.7,
    "Termination Rights": 0.7,
    "Third Party Beneficiary": 0.6,
    "Uncategorized": 0.5,  # Default moderate risk for unknown clauses
}

def get_risk_score(clause_type):
    """Get the risk score for a given clause type."""
    return RISK_MAP.get(clause_type, 0.5)  # Default to 0.5 if type not found

def get_risk_explanation(clause_type, risk_score):
    """Generate a simple explanation for the risk score."""
    explanations = {
        "Indemnification": "Indemnification clauses expose parties to financial liability for third-party claims.",
        "Limitation of Liability": "Limitation of Liability clauses cap potential damages and may exclude certain types of damages.",
        "Confidentiality": "Confidentiality provisions restrict information sharing and may impose burdens on information handling.",
        "Non-Compete": "Non-compete clauses restrict business activities and may impact future opportunities.",
        "Termination": "Termination provisions define how parties can end the agreement and associated obligations.", 
    }
    
    # Generate risk level description
    if risk_score <= 0.3:
        risk_level = "low risk"
    elif risk_score <= 0.6:
        risk_level = "moderate risk"
    elif risk_score <= 0.9:
        risk_level = "high risk"
    else:
        risk_level = "very high risk"
    
    # Use specific explanation if available, otherwise use a generic one
    for key in explanations.keys():
        if key in clause_type:
            return f"This {clause_type} clause presents {risk_level}. {explanations[key]}"
    
    return f"This {clause_type} clause presents {risk_level}. Review by legal counsel is recommended."