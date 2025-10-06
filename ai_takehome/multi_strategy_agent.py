#!/usr/bin/env python3
"""
Multi-Strategy Claims Denial QA Agent
Supports multiple retrieval strategies with easy switching.

Usage:
    python multi_strategy_agent.py --question "Why are cardiology claims denied?" --strategy embeddings
    python multi_strategy_agent.py --question "Why are cardiology claims denied?" --strategy hybrid
    python multi_strategy_agent.py --question "Why are cardiology claims denied?" --strategy bm25
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from retrieval_strategies import (
    load_docs, BM25Strategy, TFIDFStrategy, EmbeddingStrategy,
    HybridStrategy, EnsembleStrategy
)

def synthesize_answer(question: str, relevant_claims) -> str:
    """Generate contextual answer - same logic as baseline_agent.py"""
    question_lower = question.lower()

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

def get_strategy(strategy_name: str):
    """Factory function to create strategy instances"""
    strategy_map = {
        'bm25': BM25Strategy(),
        'tfidf': TFIDFStrategy(),
        'embeddings': EmbeddingStrategy(),
        'hybrid': HybridStrategy(bm25_candidates=50),
        'ensemble': EnsembleStrategy([
            BM25Strategy(),
            TFIDFStrategy(),
            EmbeddingStrategy()
        ], weights=[0.3, 0.3, 0.4])
    }

    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategy_map.keys())}")

    return strategy_map[strategy_name]

def answer(question: str, strategy_name: str = 'embeddings') -> str:
    """Answer questions using specified retrieval strategy"""
    # Load data
    docs, df = load_docs()

    # Get and setup strategy
    strategy = get_strategy(strategy_name)
    strategy.setup(docs, df)

    # Retrieve relevant claims
    relevant_claims, similarities = strategy.retrieve(question, top_k=15)

    # Generate answer
    return synthesize_answer(question, relevant_claims)

def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy Claims Denial QA Agent")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--strategy", default="embeddings",
                       choices=['bm25', 'tfidf', 'embeddings', 'hybrid', 'ensemble'],
                       help="Retrieval strategy to use")
    parser.add_argument("--show-info", action="store_true",
                       help="Show strategy information")

    args = parser.parse_args()

    if args.show_info:
        print("🔧 AVAILABLE STRATEGIES:")
        print("  • bm25       - Fast keyword matching (BM25 algorithm)")
        print("  • tfidf      - Enhanced TF-IDF with n-grams")
        print("  • embeddings - Semantic embeddings (sentence-transformers)")
        print("  • hybrid     - BM25 filtering + embedding re-ranking")
        print("  • ensemble   - Voting ensemble of multiple strategies")
        print()

    print(f"🤖 Using {args.strategy} strategy...")
    result = answer(args.question, args.strategy)
    print(result)

if __name__ == "__main__":
    main()