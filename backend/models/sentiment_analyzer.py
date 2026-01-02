"""
Sentiment analysis for social media discussions about LLMs.

This module analyzes sentiment of collected text using NLP models.
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

# Try to import VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VADER not available. Install with: pip install vaderSentiment")

# Try to import transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    DEVICE = 0 if torch.cuda.is_available() else -1
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DEVICE = -1
    print("Warning: Transformers not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Topic keywords
TOPIC_KEYWORDS = {
    'speed': ['fast', 'slow', 'speed', 'latency', 'response time', 'quick', 'slow'],
    'accuracy': ['accurate', 'correct', 'wrong', 'error', 'mistake', 'precise', 'hallucination'],
    'cost': ['expensive', 'cheap', 'cost', 'price', 'affordable', 'pricing', 'free'],
    'usefulness': ['useful', 'helpful', 'useless', 'practical', 'effective', 'worthless'],
    'quality': ['good', 'bad', 'quality', 'excellent', 'poor', 'great', 'terrible'],
    'ease_of_use': ['easy', 'hard', 'difficult', 'simple', 'complex', 'user-friendly'],
    'reliability': ['reliable', 'unstable', 'crash', 'stable', 'dependable', 'buggy']
}


class SentimentAnalyzer:
    """Analyzer for sentiment in social media discussions."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        method: str = 'vader'
    ) -> None:
        """
        Initialize the sentiment analyzer.

        Args:
            db_manager: Optional database manager instance
            method: Analysis method ('vader' or 'transformers')
        """
        self.db_manager = db_manager or DatabaseManager()
        self.method = method
        
        # Initialize analyzer
        if method == 'vader' and VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
            logger.info("Initialized VADER sentiment analyzer")
        elif method == 'transformers' and TRANSFORMERS_AVAILABLE:
            try:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=DEVICE
                )
                logger.info(f"Initialized Transformers sentiment analyzer (device: {DEVICE})")
            except Exception as e:
                logger.warning(f"Failed to load transformers model: {e}. Falling back to VADER.")
                if VADER_AVAILABLE:
                    self.analyzer = SentimentIntensityAnalyzer()
                    self.method = 'vader'
                else:
                    raise ValueError("No sentiment analyzer available")
        else:
            if VADER_AVAILABLE:
                self.analyzer = SentimentIntensityAnalyzer()
                self.method = 'vader'
                logger.info("Using VADER as fallback")
            else:
                raise ValueError("No sentiment analyzer available. Install vaderSentiment or transformers")

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment, score, and confidence
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            }
        
        # Clean text
        text = self._clean_text(text)
        
        if self.method == 'vader':
            return self._analyze_vader(text)
        else:
            return self._analyze_transformers(text)

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text.strip()

    def _analyze_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER."""
        scores = self.analyzer.polarity_scores(text)
        
        # Determine sentiment
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 'positive'
            score = compound
            confidence = scores['pos']
        elif compound <= -0.05:
            sentiment = 'negative'
            score = compound
            confidence = scores['neg']
        else:
            sentiment = 'neutral'
            score = compound
            confidence = scores['neu']
        
        return {
            'sentiment': sentiment,
            'score': float(score),
            'confidence': float(confidence)
        }

    def _analyze_transformers(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Transformers."""
        # Truncate if too long (model limit)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = self.analyzer(text)[0]
        
        label = result['label'].lower()
        score = result['score']
        
        # Map labels
        if 'positive' in label:
            sentiment = 'positive'
            sentiment_score = score
        elif 'negative' in label:
            sentiment = 'negative'
            sentiment_score = -score
        else:
            sentiment = 'neutral'
            sentiment_score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': float(sentiment_score),
            'confidence': float(score)
        }

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts efficiently.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for text in texts:
            try:
                result = self.analyze_sentiment(text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error analyzing text: {e}")
                results.append({
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0
                })
        
        return results

    def extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics/entities from text.

        Args:
            text: Text to analyze

        Returns:
            List of topics found
        """
        if not text:
            return []
        
        text_lower = text.lower()
        topics_found = []
        
        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics_found.append(topic)
                    break  # Only add topic once
        
        return topics_found

    def update_sentiment_scores(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Update sentiment scores for all unprocessed records.

        Args:
            batch_size: Number of records to process at once
            limit: Maximum number of records to process (None for all)

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'processed': 0,
            'updated': 0,
            'errors': 0
        }
        
        # Get unprocessed records
        query = """
            SELECT id, title, content
            FROM market_sentiment
            WHERE sentiment IS NULL OR sentiment = ''
            ORDER BY scraped_at ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        records = self.db_manager.execute_query(query)
        
        if not records:
            logger.info("No unprocessed records found")
            return stats
        
        logger.info(f"Processing {len(records)} records...")
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            for record in batch:
                try:
                    # Combine title and content
                    text = f"{record.get('title', '')} {record.get('content', '')}"
                    
                    # Analyze sentiment
                    sentiment_result = self.analyze_sentiment(text)
                    
                    # Extract topics
                    topics = self.extract_topics(text)
                    
                    # Update database
                    update_query = """
                        UPDATE market_sentiment
                        SET sentiment = %s,
                            sentiment_score = %s,
                            confidence = %s,
                            topics = %s,
                            processed_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """
                    
                    self.db_manager.execute_update(
                        update_query,
                        (
                            sentiment_result['sentiment'],
                            sentiment_result['score'],
                            sentiment_result['confidence'],
                            topics,
                            record['id']
                        )
                    )
                    
                    stats['updated'] += 1
                    stats['processed'] += 1
                    
                    if stats['processed'] % 50 == 0:
                        logger.info(f"Processed {stats['processed']} records...")
                
                except Exception as e:
                    logger.warning(f"Error processing record {record.get('id')}: {e}")
                    stats['errors'] += 1
                    stats['processed'] += 1
        
        logger.info(f"Processing complete: {stats['updated']} updated, {stats['errors']} errors")
        return stats

    def get_sentiment_summary(
        self,
        model_name: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get sentiment summary for a model or all models.

        Args:
            model_name: Optional model name to filter
            days: Number of days to look back

        Returns:
            Dictionary with sentiment statistics
        """
        query = """
            SELECT 
                sentiment,
                COUNT(*) as count,
                AVG(sentiment_score) as avg_score,
                AVG(confidence) as avg_confidence
            FROM market_sentiment
            WHERE posted_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND sentiment IS NOT NULL
        """
        
        params = [days]
        
        if model_name:
            query += " AND model_name = %s"
            params.append(model_name)
        
        query += " GROUP BY sentiment"
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        summary = {
            'total': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'avg_sentiment_score': 0.0
        }
        
        for row in results:
            sentiment = row['sentiment']
            count = row['count']
            summary['total'] += count
            summary[sentiment] = count
        
        # Calculate overall sentiment score
        if results:
            weighted_sum = sum(r['count'] * r['avg_score'] for r in results)
            total = sum(r['count'] for r in results)
            if total > 0:
                summary['avg_sentiment_score'] = weighted_sum / total
        
        return summary


if __name__ == "__main__":
    """Main entry point for sentiment analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze sentiment of social media posts')
    parser.add_argument('--method', choices=['vader', 'transformers'],
                        default='vader', help='Sentiment analysis method')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for processing')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of records to process')
    parser.add_argument('--process-all', action='store_true',
                        help='Process all unprocessed records (same as --limit with no limit)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Filter by model name')
    parser.add_argument('--summary', action='store_true',
                        help='Show sentiment summary instead of processing')
    parser.add_argument('--days', type=int, default=30,
                        help='Days to look back for summary')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sentiment Analyzer")
    print("=" * 60)
    print()
    
    db = DatabaseManager()
    analyzer = SentimentAnalyzer(db_manager=db, method=args.method)
    
    try:
        if args.summary:
            # Show summary
            summary = analyzer.get_sentiment_summary(
                model_name=args.model_name,
                days=args.days
            )
            
            print("Sentiment Summary:")
            print(f"  Total posts: {summary['total']}")
            print(f"  Positive: {summary['positive']} ({summary['positive']/summary['total']*100:.1f}%)" if summary['total'] > 0 else "  Positive: 0")
            print(f"  Negative: {summary['negative']} ({summary['negative']/summary['total']*100:.1f}%)" if summary['total'] > 0 else "  Negative: 0")
            print(f"  Neutral: {summary['neutral']} ({summary['neutral']/summary['total']*100:.1f}%)" if summary['total'] > 0 else "  Neutral: 0")
            print(f"  Avg Sentiment Score: {summary['avg_sentiment_score']:.3f}")
        else:
            # Process records
            limit = None if args.process_all else args.limit
            print(f"Processing sentiment analysis (method: {args.method})...")
            stats = analyzer.update_sentiment_scores(
                batch_size=args.batch_size,
                limit=limit
            )
            
            print()
            print("Processing Summary:")
            print(f"  Processed: {stats['processed']}")
            print(f"  Updated: {stats['updated']}")
            print(f"  Errors: {stats['errors']}")
        
        print()
        print("=" * 60)
        print("✅ Completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()

