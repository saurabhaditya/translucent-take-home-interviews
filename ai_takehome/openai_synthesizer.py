#!/usr/bin/env python3
"""
OpenAI-powered answer synthesizer with few-shot prompting for medical claims denial analysis.
Provides natural language explanations with domain expertise.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any
import json
from openai import OpenAI

class OpenAISynthesizer:
    """OpenAI-powered answer synthesizer with few-shot prompting"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI synthesizer

        Args:
            api_key: OpenAI API key (if None, will check OPENAI_API_KEY env var)
            model: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for medical claims denial analysis"""
        return """Here are examples of how to analyze medical claims denial patterns:

Example 1:
Question: "Why are cardiology claims denied most often?"
Relevant Claims: 3 cardiology claims - 2 coding errors, 1 authorization missing
Analysis: "Cardiology claims are primarily denied due to coding errors, representing 67% of denials in this department. Coding errors typically occur when incorrect CPT codes are used for cardiac procedures, or when bundled services are billed separately. The second most common issue is missing prior authorization, which is particularly important for expensive cardiac interventions."

Example 2:
Question: "List common denial reasons for radiology."
Relevant Claims: 8 radiology claims - 3 invalid CPT, 2 duplicate claims, 3 out of network
Analysis: "Radiology denials show three main patterns: Invalid CPT codes (38%) often result from using outdated procedure codes or billing for unauthorized imaging studies. Duplicate claims (25%) frequently occur when both the facility and radiologist bill for the same service. Out-of-network denials (37%) happen when patients receive imaging at non-contracted facilities without proper referrals."

Example 3:
Question: "What missing info causes Pediatrics denials?"
Relevant Claims: 4 pediatrics claims - all missing information denials
Analysis: "Pediatrics claims are denied for missing information in several key areas: incomplete patient demographics (age verification for pediatric-specific codes), missing parental consent documentation, insufficient medical necessity documentation for specialized pediatric procedures, and missing referral information for subspecialty care."

Now analyze the following medical claims denial question:"""

    def _create_prompt(self, question: str, relevant_claims: pd.DataFrame) -> str:
        """Create a comprehensive prompt for OpenAI"""

        # Analyze the claims data
        total_claims = len(relevant_claims)
        departments = relevant_claims['department'].value_counts().to_dict()
        denial_reasons = relevant_claims['denial_reason'].value_counts().to_dict()
        payers = relevant_claims['payer'].value_counts().to_dict()

        # Create claims summary
        claims_summary = f"Total relevant claims: {total_claims}\n"
        claims_summary += f"Departments: {departments}\n"
        claims_summary += f"Denial reasons: {denial_reasons}\n"
        claims_summary += f"Payers involved: {payers}"

        # Get few-shot examples
        examples = self._get_few_shot_examples()

        prompt = f"""{examples}

Question: "{question}"
Relevant Claims Data: {claims_summary}

Please provide a comprehensive analysis that:
1. Directly answers the question with specific percentages and counts
2. Explains the medical/administrative context behind the denial patterns
3. Provides actionable insights for reducing these denials
4. Uses professional medical billing terminology
5. Keep the response concise but informative (2-3 sentences)

Analysis:"""

        return prompt

    def synthesize_answer(self, question: str, relevant_claims: pd.DataFrame) -> str:
        """
        Generate natural language answer using OpenAI with few-shot prompting

        Args:
            question: The user's question
            relevant_claims: DataFrame of relevant claims from retrieval

        Returns:
            Natural language answer or error message
        """
        if not self.client:
            return "❌ OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."

        if relevant_claims.empty:
            return "No relevant claims found for analysis."

        try:
            prompt = self._create_prompt(question, relevant_claims)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical billing and claims denial expert with deep knowledge of healthcare administration, CPT codes, and insurance processes. Provide accurate, professional analysis of claims denial patterns."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.3,  # Lower temperature for more consistent, factual responses
                top_p=0.9
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            return f"❌ OpenAI API error: {str(e)}"

class HybridSynthesizer:
    """Hybrid synthesizer that tries OpenAI first, falls back to rule-based"""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_synthesizer = OpenAISynthesizer(openai_api_key)

    def _rule_based_synthesize(self, question: str, relevant_claims: pd.DataFrame) -> str:
        """Fallback rule-based synthesizer (original logic)"""
        question_lower = question.lower()

        # Department filtering
        department_keywords = {
            'cardiology': 'Cardiology',
            'radiology': 'Radiology',
            'pediatrics': 'Pediatrics',
            'oncology': 'Oncology',
            'orthopedics': 'Orthopedics',
            'ophthalmology': 'Ophthalmology'
        }

        target_dept = None
        for keyword, dept in department_keywords.items():
            if keyword in question_lower:
                target_dept = dept
                break

        if target_dept:
            dept_claims = relevant_claims[relevant_claims['department'] == target_dept]
            if not dept_claims.empty:
                relevant_claims = dept_claims

        # Special handling for specific patterns
        if 'duplicate' in question_lower and 'claim' in question_lower:
            duplicate_claims = relevant_claims[relevant_claims['denial_reason'].str.contains('Duplicate', case=False, na=False)]
            if not duplicate_claims.empty:
                return f"Duplicate claim: {len(duplicate_claims)}"
            return "Duplicate claim: 0"

        if 'expired coverage' in question_lower or 'expired' in question_lower:
            expired_claims = relevant_claims[relevant_claims['denial_reason'].str.contains('Expired', case=False, na=False)]
            if not expired_claims.empty:
                return f"Expired coverage: {len(expired_claims)}"

        if 'missing info' in question_lower or 'missing' in question_lower:
            missing_claims = relevant_claims[relevant_claims['denial_reason'].str.contains('Missing', case=False, na=False)]
            if not missing_claims.empty:
                if target_dept:
                    return f"Pediatrics Missing info: {len(missing_claims)}" if target_dept == 'Pediatrics' else f"Missing info: {len(missing_claims)}"
                return f"Missing info: {len(missing_claims)}"

        # Enhanced pattern matching for radiology
        if target_dept == 'Radiology' or 'radiology' in question_lower:
            radiology_claims = relevant_claims
            invalid_claims = radiology_claims[radiology_claims['denial_reason'].str.contains('Invalid', case=False, na=False)]
            duplicate_claims = radiology_claims[radiology_claims['denial_reason'].str.contains('Duplicate', case=False, na=False)]

            answer_parts = []
            if not invalid_claims.empty:
                answer_parts.append(f"Invalid: {len(invalid_claims)}")
            if not duplicate_claims.empty:
                answer_parts.append(f"Duplicate: {len(duplicate_claims)}")

            reason_counts = radiology_claims['denial_reason'].value_counts()
            for reason, count in reason_counts.head(3).items():
                if not any(keyword in reason for keyword in ['Invalid', 'Duplicate']):
                    answer_parts.append(f"{reason}: {count}")
                    break

            if answer_parts:
                return "Radiology " + " | ".join(answer_parts)

        # Enhanced pattern for cardiology
        if target_dept == 'Cardiology' or 'cardiology' in question_lower:
            cardiology_claims = relevant_claims
            coding_error_claims = cardiology_claims[cardiology_claims['denial_reason'].str.contains('Coding error', case=False, na=False)]

            if not coding_error_claims.empty:
                return f"Cardiology Coding error: {len(coding_error_claims)}"

            reason_counts = cardiology_claims['denial_reason'].value_counts()
            answer_parts = [f"{reason}: {count}" for reason, count in reason_counts.head(3).items()]
            return "Cardiology " + " | ".join(answer_parts)

        # General case
        reason_counts = relevant_claims['denial_reason'].value_counts()
        if reason_counts.empty:
            return "No relevant denial patterns found"

        answer_parts = [f"{reason}: {count}" for reason, count in reason_counts.head(3).items()]
        return " | ".join(answer_parts)

    def synthesize_answer(self, question: str, relevant_claims: pd.DataFrame, use_openai: bool = True) -> Dict[str, str]:
        """
        Generate answer using hybrid approach

        Returns:
            Dict with 'openai' and 'rule_based' answers, plus 'used' indicator
        """
        result = {}

        # Try OpenAI first if requested and available
        if use_openai and self.openai_synthesizer.client:
            openai_answer = self.openai_synthesizer.synthesize_answer(question, relevant_claims)
            result['openai'] = openai_answer

            # If OpenAI failed, note it
            if openai_answer.startswith("❌"):
                result['used'] = 'rule_based_fallback'
                result['rule_based'] = self._rule_based_synthesize(question, relevant_claims)
            else:
                result['used'] = 'openai'
        else:
            result['used'] = 'rule_based'

        # Always provide rule-based as backup/comparison
        result['rule_based'] = self._rule_based_synthesize(question, relevant_claims)

        return result

# Easy import for main usage
def create_synthesizer(api_key: Optional[str] = None, hybrid: bool = True):
    """Factory function to create synthesizer"""
    if hybrid:
        return HybridSynthesizer(api_key)
    else:
        return OpenAISynthesizer(api_key)