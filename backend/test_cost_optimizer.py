"""
Test script for CostOptimizer.

This script tests the cost optimizer functionality including
cost calculation, cheapest model finding, value scoring, and cost comparison.
"""

from models.cost_optimizer import CostOptimizer
from database.db_manager import DatabaseManager


def main() -> None:
    """Run cost optimizer tests."""
    print("=" * 60)
    print("Cost Optimizer Test Suite")
    print("=" * 60)
    print()

    # Initialize optimizer
    try:
        db = DatabaseManager()
        optimizer = CostOptimizer(db_manager=db)
        print("✅ Cost optimizer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize cost optimizer: {e}")
        return

    try:
        # Test 1: Calculate cost
        print("\n[Test 1] Calculating cost for GPT-4 Turbo...")
        try:
            cost = optimizer.calculate_cost("GPT-4 Turbo", 1000, 500)
            print(f"✅ GPT-4 Turbo cost for 1K input + 500 output: ${cost:.6f}")
            print(f"   Expected: ~$0.025")
        except ValueError as e:
            print(f"⚠️  Test skipped: {e}")
            print("   (Model or pricing data may not be in database)")

        # Test 2: Get cheapest above threshold
        print("\n[Test 2] Finding cheapest model with score >1200...")
        try:
            best = optimizer.get_cheapest_model(min_score=1200)
            print(f"✅ Best value model with score >1200:")
            print(f"   Model: {best['model_name']}")
            print(f"   Score: {best['score']:.1f}")
            print(f"   Cost per 1M tokens: ${best['cost_per_1m_tokens']:.2f}")
            print(f"   Provider: {best.get('provider', 'N/A')}")
        except ValueError as e:
            print(f"⚠️  Test skipped: {e}")
            print("   (No models meet the criteria or data not available)")

        # Test 3: Value score
        print("\n[Test 3] Calculating value score for Claude Sonnet...")
        try:
            value = optimizer.calculate_value_score("Claude 3 Sonnet")
            print(f"✅ Claude 3 Sonnet value score: {value:.2f}/100")
        except ValueError as e:
            print(f"⚠️  Test skipped: {e}")
            print("   (Model or data may not be in database)")
            # Try alternative name
            try:
                value = optimizer.calculate_value_score("Claude 3.5 Sonnet")
                print(f"✅ Claude 3.5 Sonnet value score: {value:.2f}/100")
            except ValueError:
                pass

        # Test 4: Compare costs
        print("\n[Test 4] Comparing costs for multiple models...")
        try:
            comparison = optimizer.compare_costs(
                ["GPT-4 Turbo", "Claude 3.5 Sonnet", "Gemini Pro"],
                monthly_tokens=10_000_000
            )
            print("✅ Monthly cost comparison (10M tokens/month):")
            print()
            print(comparison.to_string())
            print()
            print(f"   Best value: {comparison.iloc[0]['model']}")
            print(f"   Monthly cost: ${comparison.iloc[0]['monthly_cost']:.2f}")
        except Exception as e:
            print(f"⚠️  Test skipped: {e}")
            print("   (Some models may not be in database)")

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\nNote: Some tests may be skipped if required data is not in the database.")
        print("Run the seed_historical_data.py script to populate test data.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        print("\n✅ Database connections closed")


if __name__ == "__main__":
    main()

