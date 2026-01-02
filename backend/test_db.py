"""
Test script for DatabaseManager.

This script tests the database manager functionality including
model insertion, arena score insertion, and data retrieval.
"""

from database.db_manager import DatabaseManager
from datetime import datetime


def main() -> None:
    """Run database manager tests."""
    print("=" * 50)
    print("Database Manager Test Suite")
    print("=" * 50)
    print()

    # Initialize database manager
    try:
        db = DatabaseManager()
        print("✅ Database manager initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database manager: {e}")
        return

    try:
        # Test 1: Insert a model
        print("\n[Test 1] Inserting model...")
        model_id = db.insert_model("GPT-4", "OpenAI")
        print(f"✅ Model inserted with ID: {model_id}")

        # Test 2: Insert arena score
        print("\n[Test 2] Inserting arena score...")
        score_id = db.insert_arena_score(
            model_id, 
            1250.0, 
            1, 
            "overall", 
            datetime.now().isoformat()  # Convert datetime to ISO format string
        )
        print(f"✅ Arena score inserted with ID: {score_id}")

        # Test 3: Get models
        print("\n[Test 3] Retrieving all models...")
        models = db.get_models()
        print(f"✅ Retrieved {len(models)} models")
        if models:
            print(f"   First model: {models[0].get('name', 'N/A')} (ID: {models[0].get('id', 'N/A')})")

        # Test 4: Get history
        print("\n[Test 4] Retrieving arena history...")
        history = db.get_arena_history(model_id)
        print(f"✅ Retrieved {len(history)} history records")
        if history:
            print(f"   Latest record: Rank {history[0].get('rank_position', 'N/A')}, "
                  f"Score {history[0].get('elo_rating', 'N/A')}")

        # Test 5: Insert pricing
        print("\n[Test 5] Inserting pricing data...")
        pricing_id = db.insert_pricing(
            model_id,
            0.01,  # input price per 1K tokens
            0.03,  # output price per 1K tokens
            datetime.now().date().isoformat(),
            "OpenAI",
            "GPT-4"
        )
        print(f"✅ Pricing data inserted with ID: {pricing_id}")

        # Test 6: Get latest pricing
        print("\n[Test 6] Retrieving latest pricing...")
        pricing = db.get_latest_pricing()
        print(f"✅ Retrieved {len(pricing)} pricing records")
        if pricing:
            print(f"   First record: {pricing[0].get('model_name', 'N/A')} - "
                  f"Input: ${pricing[0].get('input_cost_per_token', 0):.4f}, "
                  f"Output: ${pricing[0].get('output_cost_per_token', 0):.4f}")

        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        db.close()
        print("\n✅ Database connections closed")


if __name__ == "__main__":
    main()

