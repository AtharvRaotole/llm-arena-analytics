"""
Sentiment scraper for social media discussions about LLMs.

This module scrapes Reddit, Hacker News, and Twitter for discussions
about LLM models and stores sentiment data.
"""

import time
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta, date
import requests
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

# Try to import PRAW for Reddit
try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    print("Warning: PRAW not available. Install with: pip install praw")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Major LLM model names to search for
MODEL_KEYWORDS = [
    'GPT-4', 'GPT-3.5', 'GPT-4 Turbo', 'GPT-4o',
    'Claude', 'Claude 3', 'Claude 3.5', 'Claude Opus', 'Claude Sonnet',
    'Gemini', 'Gemini Pro', 'Gemini Ultra',
    'Llama', 'Llama 2', 'Llama 3',
    'Mistral', 'Mistral Large',
    'Palm', 'Bard', 'Jurassic', 'Cohere'
]


class SentimentScraper:
    """Scraper for social media sentiment about LLMs."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: str = "LLM-Arena-Analytics/1.0"
    ) -> None:
        """
        Initialize the sentiment scraper.

        Args:
            db_manager: Optional database manager instance
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
        """
        self.db_manager = db_manager or DatabaseManager()
        self.reddit = None
        self.scraped_ids: Set[str] = set()
        
        # Initialize Reddit if credentials provided
        if REDDIT_AVAILABLE and reddit_client_id and reddit_client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
                logger.info("Reddit API initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit: {e}")

    def extract_model_mentions(self, text: str) -> List[str]:
        """
        Extract model names mentioned in text.

        Args:
            text: Text to analyze

        Returns:
            List of model names found
        """
        if not text:
            return []
        
        text_lower = text.lower()
        mentioned_models = []
        
        for keyword in MODEL_KEYWORDS:
            if keyword.lower() in text_lower:
                mentioned_models.append(keyword)
        
        return mentioned_models

    def scrape_reddit(
        self,
        subreddits: List[str],
        days: int = 7,
        limit_per_sub: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Scrape Reddit for LLM discussions.

        Args:
            subreddits: List of subreddit names
            days: Number of days to look back
            limit_per_sub: Maximum posts per subreddit

        Returns:
            List of post dictionaries
        """
        if not self.reddit:
            logger.warning("Reddit API not available")
            return []
        
        posts = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for subreddit_name in subreddits:
            try:
                logger.info(f"Scraping r/{subreddit_name}...")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Rate limiting: 60 requests/minute
                time.sleep(1)  # Conservative delay
                
                for submission in subreddit.new(limit=limit_per_sub):
                    # Check date
                    post_date = datetime.fromtimestamp(submission.created_utc)
                    if post_date < cutoff_date:
                        break
                    
                    # Extract model mentions
                    title_text = submission.title
                    selftext = submission.selftext if hasattr(submission, 'selftext') else ""
                    full_text = f"{title_text} {selftext}"
                    
                    mentioned_models = self.extract_model_mentions(full_text)
                    
                    if mentioned_models:
                        post_data = {
                            'model_name': mentioned_models[0],  # Use first mentioned
                            'source': 'reddit',
                            'post_id': submission.id,
                            'title': title_text,
                            'content': full_text[:5000],  # Limit content length
                            'author': str(submission.author) if submission.author else 'Unknown',
                            'score': submission.score,
                            'url': f"https://reddit.com{submission.permalink}",
                            'posted_at': post_date.isoformat()
                        }
                        posts.append(post_data)
                
                logger.info(f"Found {len([p for p in posts if p['source'] == 'reddit'])} relevant posts in r/{subreddit_name}")
                
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {e}")
                continue
        
        return posts

    def scrape_hackernews(
        self,
        days: int = 7,
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Scrape Hacker News for LLM discussions.

        Args:
            days: Number of days to look back
            limit: Maximum posts to fetch

        Returns:
            List of post dictionaries
        """
        posts = []
        cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
        
        try:
            logger.info("Scraping Hacker News...")
            
            # HN Algolia API
            url = "https://hn.algolia.com/api/v1/search_by_date"
            params = {
                'query': 'LLM OR GPT OR Claude OR Gemini',
                'tags': 'story',
                'numericFilters': f'created_at_i>{cutoff_timestamp}',
                'hitsPerPage': min(limit, 1000)
            }
            
            # Rate limiting: 1 request/second
            time.sleep(1)
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for hit in data.get('hits', [])[:limit]:
                title = hit.get('title', '')
                story_text = hit.get('story_text', '') or hit.get('comment_text', '')
                full_text = f"{title} {story_text}"
                
                # Extract model mentions
                mentioned_models = self.extract_model_mentions(full_text)
                
                if mentioned_models:
                    post_data = {
                        'model_name': mentioned_models[0],
                        'source': 'hackernews',
                        'post_id': str(hit.get('objectID', '')),
                        'title': title,
                        'content': full_text[:5000],
                        'author': hit.get('author', 'Unknown'),
                        'score': hit.get('points', 0),
                        'url': hit.get('url', f"https://news.ycombinator.com/item?id={hit.get('objectID')}"),
                        'posted_at': datetime.fromtimestamp(hit.get('created_at_i', 0)).isoformat()
                    }
                    posts.append(post_data)
            
            logger.info(f"Found {len(posts)} relevant posts on Hacker News")
            
        except Exception as e:
            logger.error(f"Error scraping Hacker News: {e}")
        
        return posts

    def scrape_twitter(
        self,
        days: int = 7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Scrape Twitter/X for LLM discussions.

        Note: Requires Twitter API v2 access. Returns empty list if not available.

        Args:
            days: Number of days to look back
            limit: Maximum tweets to fetch

        Returns:
            List of tweet dictionaries
        """
        # Twitter API integration would go here
        # Requires Twitter API v2 credentials
        logger.info("Twitter scraping not implemented (requires API access)")
        return []

    def store_sentiment(
        self,
        posts: List[Dict[str, Any]]
    ) -> int:
        """
        Store sentiment data in database.

        Args:
            posts: List of post dictionaries

        Returns:
            Number of records inserted
        """
        if not posts:
            return 0
        
        inserted = 0
        
        for post in posts:
            try:
                # Check if already exists
                check_query = """
                    SELECT id FROM market_sentiment
                    WHERE source = %s AND post_id = %s
                """
                existing = self.db_manager.execute_query(
                    check_query,
                    (post['source'], post['post_id'])
                )
                
                if existing:
                    continue  # Skip duplicates
                
                # Insert
                insert_query = """
                    INSERT INTO market_sentiment
                    (model_name, source, post_id, title, content, author, score, url, posted_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source, post_id) DO NOTHING
                """
                
                self.db_manager.execute_update(
                    insert_query,
                    (
                        post.get('model_name'),
                        post['source'],
                        post['post_id'],
                        post.get('title'),
                        post['content'],
                        post.get('author'),
                        post.get('score', 0),
                        post.get('url'),
                        post['posted_at']
                    )
                )
                inserted += 1
                
            except Exception as e:
                logger.warning(f"Failed to store post {post.get('post_id')}: {e}")
                continue
        
        return inserted

    def scrape_all_sources(
        self,
        days: int = 7,
        reddit_subreddits: Optional[List[str]] = None,
        reddit_limit: int = 100
    ) -> Dict[str, int]:
        """
        Scrape all available sources.

        Args:
            days: Number of days to look back
            reddit_subreddits: List of subreddits to scrape
            reddit_limit: Limit per subreddit

        Returns:
            Dictionary with scraping statistics
        """
        if reddit_subreddits is None:
            reddit_subreddits = ['LocalLLaMA', 'MachineLearning', 'artificial', 'ChatGPT']
        
        stats = {
            'reddit_posts': 0,
            'hackernews_posts': 0,
            'twitter_posts': 0,
            'total_stored': 0
        }
        
        all_posts = []
        
        # Scrape Reddit
        if self.reddit:
            reddit_posts = self.scrape_reddit(
                reddit_subreddits,
                days=days,
                limit_per_sub=reddit_limit
            )
            all_posts.extend(reddit_posts)
            stats['reddit_posts'] = len(reddit_posts)
        
        # Scrape Hacker News
        hn_posts = self.scrape_hackernews(days=days, limit=200)
        all_posts.extend(hn_posts)
        stats['hackernews_posts'] = len(hn_posts)
        
        # Scrape Twitter (if available)
        twitter_posts = self.scrape_twitter(days=days, limit=100)
        all_posts.extend(twitter_posts)
        stats['twitter_posts'] = len(twitter_posts)
        
        # Store in database
        stored = self.store_sentiment(all_posts)
        stats['total_stored'] = stored
        
        return stats


if __name__ == "__main__":
    """Main entry point for sentiment scraper."""
    parser = argparse.ArgumentParser(description='Scrape social media sentiment about LLMs')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back')
    parser.add_argument('--reddit-client-id', type=str, help='Reddit API client ID')
    parser.add_argument('--reddit-client-secret', type=str, help='Reddit API client secret')
    parser.add_argument('--reddit-subreddits', nargs='+', 
                        default=['LocalLLaMA', 'MachineLearning'],
                        help='Subreddits to scrape')
    parser.add_argument('--reddit-limit', type=int, default=100,
                        help='Posts per subreddit')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sentiment Scraper")
    print("=" * 60)
    print()
    
    # Initialize
    db = DatabaseManager()
    scraper = SentimentScraper(
        db_manager=db,
        reddit_client_id=args.reddit_client_id,
        reddit_client_secret=args.reddit_client_secret
    )
    
    try:
        # Scrape all sources
        print(f"Scraping last {args.days} days...")
        stats = scraper.scrape_all_sources(
            days=args.days,
            reddit_subreddits=args.reddit_subreddits,
            reddit_limit=args.reddit_limit
        )
        
        print()
        print("Scraping Summary:")
        print(f"  Reddit posts found: {stats['reddit_posts']}")
        print(f"  Hacker News posts found: {stats['hackernews_posts']}")
        print(f"  Twitter posts found: {stats['twitter_posts']}")
        print(f"  Total stored: {stats['total_stored']}")
        print()
        print("=" * 60)
        print("✅ Scraping completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()

