"""
Test script for Flight Price Forecast LLM Agent with Google Gemini
Run this after setting up your GEMINI_API_KEY
"""

from llm_agent import FlightLLMAgent
from app import FlightPricePredictor
import sys
import os

# Add parent directory to path to import from ui folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ui'))


def test_agent():
    """Test the LLM agent with sample queries"""

    print("=" * 60)
    print("Flight Price Forecast - AI Assistant Test (Google Gemini)")
    print("=" * 60)
    print()

    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print()
        print("To set it up:")
        print("1. Get your free API key: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable:")
        print("   $env:GEMINI_API_KEY = 'your-key-here'")
        print()
        print("See GEMINI_SETUP.md for detailed instructions")
        return

    # Initialize predictor and agent
    print("Initializing...")
    predictor = FlightPricePredictor()

    try:
        agent = FlightLLMAgent(predictor)
        print("✓ Agent initialized successfully with Google Gemini!")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure you have:")
        print("1. Installed: pip install google-generativeai")
        print("2. Set GEMINI_API_KEY environment variable")
        print()
        print("See GEMINI_SETUP.md for setup instructions")
        return

    # Test queries
    test_queries = [
        "What's the price for a flight from New York to Los Angeles?",
        "When is the best time to book a flight?",
        "Should I fly on Tuesday or Friday?",
        "What are some tips for finding cheap flights?"
    ]

    print("Running test queries...")
    print("=" * 60)
    print()

    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}")
        print(f"User: {query}")
        print()

        try:
            response = agent.chat(query)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Error: {e}")

        print()
        print("-" * 60)
        print()

    print("=" * 60)
    print("Test complete!")
    print()
    print("To test interactively:")
    print("1. Run: python ui/app.py")
    print("2. Open: http://localhost:5000")
    print("3. Click the purple chat button")
    print("=" * 60)


if __name__ == "__main__":
    test_agent()
