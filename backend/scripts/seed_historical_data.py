"""
Script to seed historical data for LLM Arena Analytics.

Generates realistic historical arena scores and pricing data for the past 90 days.
"""

import argparse
import sys
import random
import math
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager


# Top models with base scores and providers
TOP_MODELS = [
    {'name': 'GPT-4 Turbo', 'provider': 'OpenAI', 'base_score': 1250, 'trend': 0.15},
    {'name': 'Claude 3.5 Sonnet', 'provider': 'Anthropic', 'base_score': 1248, 'trend': 0.20},
    {'name': 'GPT-4', 'provider': 'OpenAI', 'base_score': 1235, 'trend': 0.10},
    {'name': 'Claude 3 Opus', 'provider': 'Anthropic', 'base_score': 1230, 'trend': 0.12},
    {'name': 'Gemini Pro', 'provider': 'Google', 'base_score': 1220, 'trend': 0.18},
    {'name': 'Llama 3 70B', 'provider': 'Meta', 'base_score': 1210, 'trend': 0.25},
    {'name': 'Mistral Large', 'provider': 'Mistral AI', 'base_score': 1200, 'trend': 0.22},
    {'name': 'GPT-3.5 Turbo', 'provider': 'OpenAI', 'base_score': 1180, 'trend': 0.08},
    {'name': 'Claude 3 Sonnet', 'provider': 'Anthropic', 'base_score': 1190, 'trend': 0.15},
    {'name': 'Gemini Ultra', 'provider': 'Google', 'base_score': 1205, 'trend': 0.16},
]

# Base pricing data (per 1K tokens)
BASE_PRICING = {
    'GPT-4 Turbo': {'input': 0.01, 'output': 0.03},
    'Claude 3.5 Sonnet': {'input': 0.003, 'output': 0.015},
    'GPT-4': {'input': 0.03, 'output': 0.06},
    'Claude 3 Opus': {'input': 0.015, 'output': 0.075},
    'Gemini Pro': {'input': 0.00025, 'output': 0.001},
    'Llama 3 70B': {'input': 0.0, 'output': 0.0},  # Open source
    'Mistral Large': {'input': 0.002, 'output': 0.006},
    'GPT-3.5 Turbo': {'input': 0.0015, 'output': 0.002},
    'Claude 3 Sonnet': {'input': 0.003, 'output': 0.015},
    'Gemini Ultra': {'input': 0.0005, 'output': 0.0015},
}

# Price change events (model, date_offset, new_input, new_output)
PRICE_CHANGES = [
    ('GPT-4 Turbo', 60, 0.01, 0.03),  # Price drop 30 days ago
    ('Claude 3.5 Sonnet', 45, 0.003, 0.015),  # Price adjustment 45 days ago
    ('Gemini Pro', 30, 0.00025, 0.001),  # Price update 30 days ago
]


def generate_arena_score(
    base_score: float,
    day: int,
    trend: float,
    variance: float = 20.0
) -> Tuple[float, float, int]:
    """
    Generate realistic arena score with random walk, trend, and seasonality.

    Args:
        base_score: Base ELO score
        day: Day number (0 = start date)
        trend: Daily trend (positive = improving)
        variance: Maximum daily change

    Returns:
        Tuple of (elo_rating, win_rate, total_battles)
    """
    # Random walk component
    random_walk = random.gauss(0, variance / 3)
    
    # Trend component (gradual improvement)
    trend_component = trend * day
    
    # Seasonality (weekly pattern - slightly better on weekends)
    day_of_week = day % 7
    seasonality = 2 * math.sin(2 * math.pi * day_of_week / 7)
    
    # Calculate score
    score = base_score + random_walk + trend_component + seasonality
    
    # Ensure score stays within reasonable bounds
    score = max(1000, min(1500, score))
    
    # Calculate win rate based on score (higher score = higher win rate)
    # Win rate roughly correlates with ELO: 50% at 1200, scales with score
    win_rate = 0.5 + (score - 1200) / 2000
    win_rate = max(0.3, min(0.9, win_rate))
    
    # Total battles increases over time
    base_battles = 1000
    battles_growth = day * 50
    total_battles = int(base_battles + battles_growth + random.randint(-100, 100))
    total_battles = max(500, total_battles)
    
    return round(score, 1), round(win_rate, 3), total_battles


def get_pricing_for_date(
    model_name: str,
    target_date: date,
    start_date: date
) -> Tuple[float, float]:
    """
    Get pricing for a model on a specific date.

    Args:
        model_name: Name of the model
        target_date: Date to get pricing for
        start_date: Start date of historical data

    Returns:
        Tuple of (input_price, output_price)
    """
    base_pricing = BASE_PRICING.get(model_name, {'input': 0.0, 'output': 0.0})
    input_price = base_pricing['input']
    output_price = base_pricing['output']
    
    # Check for price changes
    days_since_start = (target_date - start_date).days
    
    for model, days_offset, new_input, new_output in PRICE_CHANGES:
        if model == model_name and days_since_start >= days_offset:
            input_price = new_input
            output_price = new_output
    
    return input_price, output_price


def insert_historical_arena_data(
    db: DatabaseManager,
    model_id: int,
    model_name: str,
    start_date: date,
    end_date: date,
    base_score: float,
    trend: float
) -> int:
    """
    Insert historical arena data for a model.

    Args:
        db: Database manager instance
        model_id: ID of the model
        model_name: Name of the model
        start_date: Start date for historical data
        end_date: End date for historical data
        base_score: Base ELO score
        trend: Daily trend

    Returns:
        Number of records inserted
    """
    current_date = start_date
    records_inserted = 0
    day = 0
    
    while current_date <= end_date:
        score, win_rate, total_battles = generate_arena_score(base_score, day, trend)
        
        # Calculate rank (simplified - in reality would be based on all models)
        # For now, use a simple ranking based on score
        rank = max(1, int(10 - (score - 1200) / 20) + random.randint(-1, 1))
        rank = max(1, min(20, rank))
        
        try:
            # Insert arena score
            db.insert_arena_score(
                model_id=model_id,
                score=score,
                rank=rank,
                category='overall',
                date=current_date.isoformat()
            )
            records_inserted += 1
        except Exception as e:
            print(f"Warning: Failed to insert arena score for {model_name} on {current_date}: {e}")
        
        current_date += timedelta(days=1)
        day += 1
    
    return records_inserted


def insert_historical_pricing_data(
    db: DatabaseManager,
    model_id: int,
    model_name: str,
    start_date: date,
    end_date: date
) -> int:
    """
    Insert historical pricing data for a model.

    Args:
        db: Database manager instance
        model_id: ID of the model
        model_name: Name of the model
        start_date: Start date for historical data
        end_date: End date for historical data

    Returns:
        Number of records inserted
    """
    records_inserted = 0
    current_date = start_date
    last_input_price = None
    last_output_price = None
    
    while current_date <= end_date:
        input_price, output_price = get_pricing_for_date(model_name, current_date, start_date)
        
        # Only insert if price changed or it's the first day
        if (input_price != last_input_price or 
            output_price != last_output_price or 
            current_date == start_date):
            
            try:
                # Get provider from model info
                provider = None
                for model_info in TOP_MODELS:
                    if model_info['name'] == model_name:
                        provider = model_info['provider']
                        break
                
                db.insert_pricing(
                    model_id=model_id,
                    input_price=input_price,
                    output_price=output_price,
                    date=current_date.isoformat(),
                    provider=provider,
                    model_name=model_name
                )
                records_inserted += 1
                last_input_price = input_price
                last_output_price = output_price
            except Exception as e:
                print(f"Warning: Failed to insert pricing for {model_name} on {current_date}: {e}")
        
        current_date += timedelta(days=1)
    
    return records_inserted


def seed_historical_data(
    db: DatabaseManager,
    start_date: date,
    end_date: date,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Seed historical data for all top models.

    Args:
        db: Database manager instance
        start_date: Start date for historical data
        end_date: End date for historical data
        dry_run: If True, don't actually insert data

    Returns:
        Dictionary with statistics
    """
    stats = {
        'models_processed': 0,
        'arena_records': 0,
        'pricing_records': 0,
        'errors': []
    }
    
    print(f"Seeding historical data from {start_date} to {end_date}")
    if dry_run:
        print("DRY RUN MODE: No data will be inserted")
    print()
    
    for model_info in TOP_MODELS:
        model_name = model_info['name']
        provider = model_info['provider']
        base_score = model_info['base_score']
        trend = model_info['trend']
        
        print(f"Processing {model_name} ({provider})...")
        
        try:
            # Insert or get model
            if not dry_run:
                model_id = db.insert_model(
                    name=model_name,
                    provider=provider
                )
            else:
                model_id = 1  # Placeholder for dry run
                print(f"  [DRY RUN] Would insert model: {model_name}")
            
            # Insert arena data
            if not dry_run:
                arena_count = insert_historical_arena_data(
                    db, model_id, model_name, start_date, end_date, base_score, trend
                )
                stats['arena_records'] += arena_count
                print(f"  Inserted {arena_count} arena records")
            else:
                days = (end_date - start_date).days + 1
                print(f"  [DRY RUN] Would insert {days} arena records")
                stats['arena_records'] += days
            
            # Insert pricing data
            if not dry_run:
                pricing_count = insert_historical_pricing_data(
                    db, model_id, model_name, start_date, end_date
                )
                stats['pricing_records'] += pricing_count
                print(f"  Inserted {pricing_count} pricing records")
            else:
                # Count price change events
                price_changes = sum(1 for m, _, _, _ in PRICE_CHANGES if m == model_name)
                print(f"  [DRY RUN] Would insert {price_changes + 1} pricing records")
                stats['pricing_records'] += price_changes + 1
            
            stats['models_processed'] += 1
            print()
            
        except Exception as e:
            error_msg = f"Error processing {model_name}: {e}"
            stats['errors'].append(error_msg)
            print(f"  ERROR: {error_msg}")
            print()
    
    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Seed historical data for LLM Arena Analytics'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD). Defaults to 90 days ago.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD). Defaults to today.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be inserted without actually inserting'
    )
    parser.add_argument(
        '--db-host',
        type=str,
        default=None,
        help='Database host (overrides DB_HOST env var)'
    )
    parser.add_argument(
        '--db-port',
        type=int,
        default=None,
        help='Database port (overrides DB_PORT env var)'
    )
    parser.add_argument(
        '--db-name',
        type=str,
        default=None,
        help='Database name (overrides DB_NAME env var)'
    )
    parser.add_argument(
        '--db-user',
        type=str,
        default=None,
        help='Database user (overrides DB_USER env var)'
    )
    parser.add_argument(
        '--db-password',
        type=str,
        default=None,
        help='Database password (overrides DB_PASSWORD env var)'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = date.today()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    else:
        start_date = end_date - timedelta(days=90)
    
    if start_date >= end_date:
        print("Error: Start date must be before end date")
        sys.exit(1)
    
    # Initialize database
    try:
        db = DatabaseManager(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password
        )
        print("Database manager initialized")
    except Exception as e:
        print(f"Failed to initialize database manager: {e}")
        sys.exit(1)
    
    try:
        # Seed data
        stats = seed_historical_data(db, start_date, end_date, args.dry_run)
        
        # Print summary
        print("=" * 60)
        print("Seeding Summary")
        print("=" * 60)
        print(f"Models processed: {stats['models_processed']}")
        print(f"Arena records: {stats['arena_records']}")
        print(f"Pricing records: {stats['pricing_records']}")
        print(f"Errors: {len(stats['errors'])}")
        
        if stats['errors']:
            print("\nErrors encountered:")
            for error in stats['errors']:
                print(f"  - {error}")
        
        print("=" * 60)
        
        if stats['errors']:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

