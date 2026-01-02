"""
Scraper pipeline for Chatbot Arena data integration.

This module orchestrates the scraping process and integrates scraped data
with the database, handling model insertion and arena score updates.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, date

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from scrapers.chatbot_arena_scraper import ChatbotArenaScraper


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScraperPipeline:
    """Pipeline for scraping and integrating Chatbot Arena data."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        dry_run: bool = False,
        force: bool = False
    ) -> None:
        """
        Initialize the scraper pipeline.

        Args:
            db_manager: Database manager instance
            dry_run: If True, don't actually insert data
            force: If True, force update even if today's data exists
        """
        self.db_manager = db_manager
        self.dry_run = dry_run
        self.force = force
        self.stats = {
            'models_scraped': 0,
            'new_models_added': 0,
            'scores_updated': 0,
            'scores_skipped': 0,
            'errors': []
        }

    def check_model_exists(self, model_name: str) -> Optional[int]:
        """
        Check if a model exists in the database.

        Args:
            model_name: Name of the model

        Returns:
            Model ID if exists, None otherwise
        """
        model = self.db_manager.get_model_by_name(model_name)
        return model['id'] if model else None

    def check_score_exists_today(self, model_id: int) -> bool:
        """
        Check if a score already exists for today.

        Args:
            model_id: ID of the model

        Returns:
            True if score exists for today, False otherwise
        """
        # Skip check for dry run placeholder
        if model_id == 0:
            return False
        
        try:
            history = self.db_manager.get_arena_history(model_id, days=1)
            
            # Check if any record exists for today
            today = date.today()
            for record in history:
                record_date = record.get('recorded_at')
                if record_date:
                    if isinstance(record_date, str):
                        try:
                            # Try parsing ISO format
                            record_date = datetime.fromisoformat(record_date.replace('Z', '+00:00'))
                            record_date = record_date.date()
                        except (ValueError, AttributeError):
                            continue
                    elif hasattr(record_date, 'date'):
                        record_date = record_date.date()
                    else:
                        continue
                    
                    if isinstance(record_date, date) and record_date == today:
                        return True
        except Exception as e:
            logger.warning(f"Error checking score existence: {e}")
            return False
        
        return False

    def process_model(
        self,
        model_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single model: insert/update model and score.

        Args:
            model_data: Model data dictionary from scraper

        Returns:
            Dictionary with processing results
        """
        model_name = model_data.get('model_name')
        if not model_name:
            error_msg = "Model data missing 'model_name'"
            self.stats['errors'].append(error_msg)
            logger.warning(error_msg)
            return {'success': False, 'error': error_msg}

        try:
            # Check if model exists
            model_id = self.check_model_exists(model_name)
            
            if model_id is None:
                # Insert new model
                if not self.dry_run:
                    model_id = self.db_manager.insert_model(
                        name=model_name,
                        provider=model_data.get('provider')
                    )
                    logger.info(f"Inserted new model: {model_name} (ID: {model_id})")
                    self.stats['new_models_added'] += 1
                else:
                    logger.info(f"[DRY RUN] Would insert new model: {model_name}")
                    model_id = 0  # Placeholder for dry run
                    self.stats['new_models_added'] += 1
            else:
                logger.debug(f"Model already exists: {model_name} (ID: {model_id})")

            # Check if score exists for today
            if not self.force:
                if self.check_score_exists_today(model_id):
                    logger.info(f"Score already exists for {model_name} today, skipping")
                    self.stats['scores_skipped'] += 1
                    return {'success': True, 'skipped': True}

            # Insert arena score
            if not self.dry_run:
                score_id = self.db_manager.insert_arena_score(
                    model_id=model_id,
                    score=model_data.get('elo_rating', 0),
                    rank=model_data.get('rank', 0),
                    category=model_data.get('category', 'overall'),
                    date=datetime.now().isoformat()
                )
                logger.info(
                    f"Inserted arena score for {model_name}: "
                    f"Rank {model_data.get('rank')}, Score {model_data.get('elo_rating')}"
                )
                self.stats['scores_updated'] += 1
            else:
                logger.info(
                    f"[DRY RUN] Would insert arena score for {model_name}: "
                    f"Rank {model_data.get('rank')}, Score {model_data.get('elo_rating')}"
                )
                self.stats['scores_updated'] += 1

            return {'success': True, 'model_id': model_id}

        except Exception as e:
            error_msg = f"Error processing model {model_name}: {e}"
            logger.error(error_msg, exc_info=True)
            self.stats['errors'].append(error_msg)
            return {'success': False, 'error': str(e)}

    def run(self) -> Dict[str, Any]:
        """
        Run the complete scraping and integration pipeline.

        Returns:
            Dictionary with pipeline statistics
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Scraper Pipeline")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.info("DRY RUN MODE: No data will be inserted")
        if self.force:
            logger.info("FORCE MODE: Will update even if today's data exists")

        try:
            # Initialize scraper
            scraper = ChatbotArenaScraper(
                delay_min=2.0,
                delay_max=3.0,
                max_retries=3
            )

            # Scrape leaderboard
            logger.info("Scraping Chatbot Arena leaderboard...")
            models = scraper.scrape_leaderboard()
            self.stats['models_scraped'] = len(models)

            if not models:
                logger.warning("No models scraped. Exiting pipeline.")
                return self.stats

            logger.info(f"Scraped {len(models)} models")

            # Process each model
            logger.info("Processing models and inserting into database...")
            for i, model_data in enumerate(models, 1):
                logger.debug(f"Processing model {i}/{len(models)}: {model_data.get('model_name')}")
                self.process_model(model_data)

            # Save to JSON file
            output_dir = "../../data"
            filepath = scraper.save_to_json(models, output_dir=output_dir)
            logger.info(f"Data saved to JSON: {filepath}")

            # Print summary
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self._print_summary(elapsed_time)

            return self.stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.stats['errors'].append(f"Pipeline error: {e}")
            return self.stats

    def _print_summary(self, elapsed_time: float) -> None:
        """
        Print pipeline execution summary.

        Args:
            elapsed_time: Total execution time in seconds
        """
        logger.info("=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"Models scraped: {self.stats['models_scraped']}")
        logger.info(f"New models added: {self.stats['new_models_added']}")
        logger.info(f"Scores updated: {self.stats['scores_updated']}")
        logger.info(f"Scores skipped: {self.stats['scores_skipped']}")
        logger.info(f"Errors: {len(self.stats['errors'])}")
        logger.info(f"Execution time: {elapsed_time:.1f}s")
        
        if self.stats['errors']:
            logger.warning("Errors encountered:")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                logger.warning(f"  - {error}")
            if len(self.stats['errors']) > 10:
                logger.warning(f"  ... and {len(self.stats['errors']) - 10} more errors")
        
        logger.info("=" * 60)


def main() -> None:
    """Main entry point for the scraper pipeline."""
    parser = argparse.ArgumentParser(
        description='Scrape Chatbot Arena leaderboard and integrate with database'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be inserted without actually inserting'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update even if today\'s data exists'
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

    # Initialize database manager
    try:
        db_manager = DatabaseManager(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password
        )
        logger.info("Database manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database manager: {e}")
        sys.exit(1)

    # Initialize and run pipeline
    pipeline = ScraperPipeline(
        db_manager=db_manager,
        dry_run=args.dry_run,
        force=args.force
    )

    try:
        stats = pipeline.run()
        
        # Exit with error code if there were errors
        if stats['errors']:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()

