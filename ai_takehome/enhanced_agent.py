#!/usr/bin/env python3
"""
Enhanced Claims Denial QA Agent with OpenAI Few-Shot Answer Synthesis
Supports multiple retrieval strategies + OpenAI-powered natural language generation.

Usage:
    # Rule-based synthesis (original)
    python enhanced_agent.py --question "Why are cardiology claims denied?" --strategy embeddings

    # OpenAI synthesis (requires API key)
    python enhanced_agent.py --question "Why are cardiology claims denied?" --strategy embeddings --use-openai

    # Compare both approaches
    python enhanced_agent.py --question "Why are cardiology claims denied?" --strategy embeddings --compare-synthesis
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from retrieval_strategies import (
    load_docs, BM25Strategy, TFIDFStrategy, EmbeddingStrategy,
    HybridStrategy, EnsembleStrategy
)
from openai_synthesizer import HybridSynthesizer

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

def answer(question: str, strategy_name: str = 'embeddings', use_openai: bool = False,
          compare_synthesis: bool = False, openai_api_key: str = None) -> str:
    """Answer questions using specified retrieval strategy and synthesis method"""

    # Load data
    docs, df = load_docs()

    # Get and setup strategy
    strategy = get_strategy(strategy_name)
    strategy.setup(docs, df)

    # Retrieve relevant claims
    relevant_claims, similarities = strategy.retrieve(question, top_k=15)

    # Create synthesizer
    synthesizer = HybridSynthesizer(openai_api_key)

    if compare_synthesis:
        # Show both approaches
        result = synthesizer.synthesize_answer(question, relevant_claims, use_openai=True)

        output = f"🔍 RETRIEVAL: {strategy.name}\n"
        output += f"📊 FOUND: {len(relevant_claims)} relevant claims\n\n"

        if 'openai' in result:
            output += f"🤖 OPENAI SYNTHESIS:\n{result['openai']}\n\n"

        output += f"📋 RULE-BASED SYNTHESIS:\n{result['rule_based']}\n"
        output += f"\n✅ USED: {result['used']}"

        return output

    elif use_openai:
        # Use OpenAI synthesis
        result = synthesizer.synthesize_answer(question, relevant_claims, use_openai=True)
        if result['used'] == 'openai':
            return result['openai']
        else:
            return f"⚠️ OpenAI unavailable, using rule-based: {result['rule_based']}"

    else:
        # Use rule-based synthesis (original)
        result = synthesizer.synthesize_answer(question, relevant_claims, use_openai=False)
        return result['rule_based']

def main():
    parser = argparse.ArgumentParser(description="Enhanced Claims Denial QA Agent with OpenAI")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--strategy", default="embeddings",
                       choices=['bm25', 'tfidf', 'embeddings', 'hybrid', 'ensemble'],
                       help="Retrieval strategy to use")
    parser.add_argument("--use-openai", action="store_true",
                       help="Use OpenAI for answer synthesis")
    parser.add_argument("--compare-synthesis", action="store_true",
                       help="Compare OpenAI vs rule-based synthesis")
    parser.add_argument("--openai-api-key",
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--show-info", action="store_true",
                       help="Show strategy and synthesis information")

    args = parser.parse_args()

    if args.show_info:
        print("🔧 AVAILABLE STRATEGIES:")
        print("  • bm25       - Fast keyword matching (BM25 algorithm)")
        print("  • tfidf      - Enhanced TF-IDF with n-grams")
        print("  • embeddings - Semantic embeddings (sentence-transformers)")
        print("  • hybrid     - BM25 filtering + embedding re-ranking")
        print("  • ensemble   - Voting ensemble of multiple strategies")
        print()
        print("🤖 SYNTHESIS OPTIONS:")
        print("  • Default    - Rule-based synthesis (fast, reliable)")
        print("  • --use-openai - OpenAI GPT synthesis (natural language)")
        print("  • --compare-synthesis - Show both approaches side by side")
        print()
        print("🔑 OPENAI SETUP:")
        print("  • Set OPENAI_API_KEY environment variable, or")
        print("  • Use --openai-api-key parameter")
        print()
        return

    # Check for OpenAI setup if needed
    if (args.use_openai or args.compare_synthesis):
        api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  OpenAI API key required for OpenAI synthesis.")
            print("   Set OPENAI_API_KEY environment variable or use --openai-api-key")
            print("   Falling back to rule-based synthesis...\n")
            args.use_openai = False
            args.compare_synthesis = False

    print(f"🤖 Using {args.strategy} strategy...")

    result = answer(
        args.question,
        args.strategy,
        args.use_openai,
        args.compare_synthesis,
        args.openai_api_key
    )

    print(result)

if __name__ == "__main__":
    main()