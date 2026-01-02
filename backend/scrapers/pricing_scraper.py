"""
Scraper for LLM pricing information from various providers.

This module scrapes pricing data from OpenAI, Anthropic, and Google AI
and returns structured pricing information.
"""

import time
import logging
import random
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import requests
from bs4 import BeautifulSoup


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PricingScraper:
    """Scraper for LLM pricing information from various providers."""

    # User-Agent rotation list
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    ]

    # Provider URLs
    PROVIDER_URLS = {
        'OpenAI': 'https://openai.com/api/pricing/',
        'Anthropic': 'https://www.anthropic.com/pricing',
        'Google': 'https://ai.google.dev/pricing'
    }

    def __init__(
        self,
        delay_min: float = 2.0,
        delay_max: float = 3.0,
        max_retries: int = 3
    ) -> None:
        """
        Initialize the pricing scraper.

        Args:
            delay_min: Minimum delay between requests in seconds
            delay_max: Maximum delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_retries = max_retries
        self.session = requests.Session()
        self._set_random_user_agent()

    def _set_random_user_agent(self) -> None:
        """Set a random User-Agent for the session."""
        user_agent = random.choice(self.USER_AGENTS)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        logger.debug(f"Set User-Agent: {user_agent}")

    def _rate_limit_delay(self) -> None:
        """Add a random delay between requests to respect rate limits."""
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"Rate limiting: waiting {delay:.2f} seconds")
        time.sleep(delay)

    def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (requests.RequestException, Exception) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    time.sleep(wait_time)
                    self._set_random_user_agent()
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise

        if last_exception:
            raise last_exception

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a web page.

        Args:
            url: URL to fetch

        Returns:
            BeautifulSoup object or None if fetch fails
        """
        try:
            logger.info(f"Fetching: {url}")
            response = self._retry_with_backoff(
                self.session.get,
                url,
                timeout=30
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            logger.debug(f"Successfully fetched and parsed page")
            return soup
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _detect_currency(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect currency and extract price.

        Args:
            text: Text containing a price

        Returns:
            Tuple of (currency_code, price_in_original_currency)
        """
        if not text:
            return None, None
        
        text = str(text).strip()
        
        # Check for free tier
        if 'free' in text.lower() or text == '0' or text == '$0':
            return 'USD', 0.0
        
        # Currency symbols and codes
        currency_map = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
            'CAD': 'CAD',
            'AUD': 'AUD',
            'EUR': 'EUR',
            'GBP': 'GBP',
            'JPY': 'JPY',
            'INR': 'INR',
            'USD': 'USD',
        }
        
        detected_currency = 'USD'  # Default
        for symbol, code in currency_map.items():
            if symbol in text:
                detected_currency = code
                break
        
        # Extract number
        price_patterns = [
            r'[\$€£¥₹]?\s*(\d+\.?\d*)\s*per\s*1k',
            r'[\$€£¥₹]?\s*(\d+\.?\d*)\s*/\s*1k',
            r'[\$€£¥₹]?\s*(\d+\.?\d*)\s*per\s*1,000',
            r'[\$€£¥₹]?\s*(\d+\.?\d*)\s*/\s*1,000',
            r'[\$€£¥₹]?\s*(\d+\.?\d*)\s*per\s*1M',
            r'[\$€£¥₹]?\s*(\d+\.?\d*)\s*/\s*1M',
            r'[\$€£¥₹]?\s*(\d+\.?\d*)\s*(?:USD|EUR|GBP|JPY|INR|CAD|AUD)?',
            r'[\$€£¥₹]?\s*(\d+\.?\d*)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1))
                    # If pattern mentions "per 1M", divide by 1000
                    if '1m' in text.lower() or '1,000,000' in text:
                        price = price / 1000
                    return detected_currency, price
                except (ValueError, IndexError):
                    continue
        
        return None, None

    def _convert_to_usd(self, amount: float, currency: str) -> float:
        """
        Convert amount from given currency to USD.

        Args:
            amount: Amount in original currency
            currency: Currency code (EUR, GBP, JPY, etc.)

        Returns:
            Amount in USD
        """
        # Simple conversion rates (should be updated regularly or fetched from API)
        # These are approximate rates - for production, use a currency API
        conversion_rates = {
            'USD': 1.0,
            'EUR': 1.08,  # 1 EUR = 1.08 USD (approximate)
            'GBP': 1.27,  # 1 GBP = 1.27 USD (approximate)
            'JPY': 0.0067,  # 1 JPY = 0.0067 USD (approximate)
            'INR': 0.012,  # 1 INR = 0.012 USD (approximate)
            'CAD': 0.74,  # 1 CAD = 0.74 USD (approximate)
            'AUD': 0.66,  # 1 AUD = 0.66 USD (approximate)
        }
        
        rate = conversion_rates.get(currency.upper(), 1.0)
        return amount * rate

    def _extract_price(self, text: str) -> Optional[float]:
        """
        Extract price from text, handling different formats and currencies.

        Args:
            text: Text containing a price

        Returns:
            Price in USD per 1K tokens, or None if not found
        """
        if not text:
            return None
        
        currency, price = self._detect_currency(text)
        
        if price is None:
            return None
        
        # Convert to USD if needed
        if currency and currency.upper() != 'USD':
            price = self._convert_to_usd(price, currency)
            logger.debug(f"Converted {price} {currency} to {price} USD")
        
        return price

    def _extract_context_window(self, text: str) -> Optional[int]:
        """
        Extract context window size from text.

        Args:
            text: Text containing context window information

        Returns:
            Context window size in tokens, or None if not found
        """
        if not text:
            return None
        
        # Look for patterns like "128k tokens", "128,000 tokens", etc.
        patterns = [
            r'(\d+(?:,\d+)?)\s*k\s*tokens?',
            r'(\d+(?:,\d+)?)\s*K\s*tokens?',
            r'(\d+(?:,\d+)?)\s*,\s*000\s*tokens?',
            r'(\d+(?:,\d+)?)\s*tokens?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(text), re.IGNORECASE)
            if match:
                try:
                    value = match.group(1).replace(',', '')
                    num = int(value)
                    # If it's a small number (< 1000), assume it's in thousands
                    if num < 1000 and 'k' in pattern.lower():
                        num = num * 1000
                    return num
                except (ValueError, IndexError):
                    continue
        
        return None

    def scrape_openai_pricing(self) -> List[Dict[str, Any]]:
        """
        Scrape OpenAI pricing information.

        Returns:
            List of dictionaries containing OpenAI pricing data
        """
        logger.info("Scraping OpenAI pricing...")
        url = self.PROVIDER_URLS['OpenAI']
        soup = self._fetch_page(url)
        
        if not soup:
            logger.error("Failed to fetch OpenAI pricing page")
            return []
        
        pricing_data = []
        timestamp = datetime.now().isoformat()
        
        try:
            # OpenAI pricing page structure varies, try multiple approaches
            # Look for tables with pricing information
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(['th', 'td'])]
                
                # Find relevant columns
                model_col = None
                input_col = None
                output_col = None
                context_col = None
                
                for i, header in enumerate(headers):
                    if 'model' in header or 'name' in header:
                        model_col = i
                    elif 'input' in header:
                        input_col = i
                    elif 'output' in header:
                        output_col = i
                    elif 'context' in header or 'window' in header:
                        context_col = i
                
                # Parse rows
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 3:
                        continue
                    
                    model_name = cells[model_col].get_text(strip=True) if model_col is not None else None
                    if not model_name:
                        continue
                    
                    # Extract prices
                    input_price = None
                    output_price = None
                    context_window = None
                    
                    if input_col is not None and input_col < len(cells):
                        input_text = cells[input_col].get_text(strip=True)
                        input_price = self._extract_price(input_text)
                    
                    if output_col is not None and output_col < len(cells):
                        output_text = cells[output_col].get_text(strip=True)
                        output_price = self._extract_price(output_text)
                    
                    if context_col is not None and context_col < len(cells):
                        context_text = cells[context_col].get_text(strip=True)
                        context_window = self._extract_context_window(context_text)
                    
                    if input_price is not None or output_price is not None:
                        pricing_data.append({
                            'provider': 'OpenAI',
                            'model_name': model_name,
                            'input_price_per_1k': input_price,
                            'output_price_per_1k': output_price,
                            'context_window': context_window,
                            'scraped_at': timestamp
                        })
            
            # Alternative: Look for structured data in script tags (JSON-LD)
            if not pricing_data:
                scripts = soup.find_all('script', type='application/ld+json')
                for script in scripts:
                    try:
                        import json
                        data = json.loads(script.string)
                        # Parse structured data if available
                        # This is provider-specific and may need adjustment
                    except:
                        pass
            
            # Fallback: Look for pricing in text content
            if not pricing_data:
                # Try to find model names and prices in the page text
                # This is a simplified approach - may need refinement
                page_text = soup.get_text()
                # Look for GPT model mentions
                gpt_models = re.findall(r'(GPT-[0-9A-Za-z\s]+)', page_text)
                for model in set(gpt_models):
                    # Try to find prices near model mentions
                    # This is a simplified extraction
                    pricing_data.append({
                        'provider': 'OpenAI',
                        'model_name': model.strip(),
                        'input_price_per_1k': None,
                        'output_price_per_1k': None,
                        'context_window': None,
                        'scraped_at': timestamp,
                        'note': 'Price extraction incomplete - manual verification needed'
                    })
            
            logger.info(f"Scraped {len(pricing_data)} OpenAI models")
            return pricing_data
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI pricing: {e}", exc_info=True)
            return []

    def scrape_anthropic_pricing(self) -> List[Dict[str, Any]]:
        """
        Scrape Anthropic pricing information.

        Returns:
            List of dictionaries containing Anthropic pricing data
        """
        logger.info("Scraping Anthropic pricing...")
        url = self.PROVIDER_URLS['Anthropic']
        soup = self._fetch_page(url)
        
        if not soup:
            logger.error("Failed to fetch Anthropic pricing page")
            return []
        
        pricing_data = []
        timestamp = datetime.now().isoformat()
        
        try:
            # Anthropic pricing page structure
            # Look for tables or structured sections
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) < 2:
                    continue
                
                # Try to identify header row
                headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(['th', 'td'])]
                
                model_col = None
                input_col = None
                output_col = None
                context_col = None
                
                for i, header in enumerate(headers):
                    if 'model' in header or 'claude' in header:
                        model_col = i
                    elif 'input' in header:
                        input_col = i
                    elif 'output' in header:
                        output_col = i
                    elif 'context' in header:
                        context_col = i
                
                # Parse data rows
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    
                    model_name = None
                    if model_col is not None and model_col < len(cells):
                        model_name = cells[model_col].get_text(strip=True)
                    elif len(cells) > 0:
                        model_name = cells[0].get_text(strip=True)
                    
                    if not model_name or 'claude' not in model_name.lower():
                        continue
                    
                    input_price = None
                    output_price = None
                    context_window = None
                    
                    if input_col is not None and input_col < len(cells):
                        input_text = cells[input_col].get_text(strip=True)
                        input_price = self._extract_price(input_text)
                    
                    if output_col is not None and output_col < len(cells):
                        output_text = cells[output_col].get_text(strip=True)
                        output_price = self._extract_price(output_text)
                    
                    if context_col is not None and context_col < len(cells):
                        context_text = cells[context_col].get_text(strip=True)
                        context_window = self._extract_context_window(context_text)
                    
                    if input_price is not None or output_price is not None:
                        pricing_data.append({
                            'provider': 'Anthropic',
                            'model_name': model_name,
                            'input_price_per_1k': input_price,
                            'output_price_per_1k': output_price,
                            'context_window': context_window,
                            'scraped_at': timestamp
                        })
            
            # Fallback: Look for Claude model mentions
            if not pricing_data:
                page_text = soup.get_text()
                claude_models = re.findall(r'(Claude\s+[0-9A-Za-z\s]+)', page_text)
                for model in set(claude_models):
                    pricing_data.append({
                        'provider': 'Anthropic',
                        'model_name': model.strip(),
                        'input_price_per_1k': None,
                        'output_price_per_1k': None,
                        'context_window': None,
                        'scraped_at': timestamp,
                        'note': 'Price extraction incomplete - manual verification needed'
                    })
            
            logger.info(f"Scraped {len(pricing_data)} Anthropic models")
            return pricing_data
            
        except Exception as e:
            logger.error(f"Error parsing Anthropic pricing: {e}", exc_info=True)
            return []

    def scrape_google_pricing(self) -> List[Dict[str, Any]]:
        """
        Scrape Google AI pricing information.

        Returns:
            List of dictionaries containing Google AI pricing data
        """
        logger.info("Scraping Google AI pricing...")
        url = self.PROVIDER_URLS['Google']
        soup = self._fetch_page(url)
        
        if not soup:
            logger.error("Failed to fetch Google AI pricing page")
            return []
        
        pricing_data = []
        timestamp = datetime.now().isoformat()
        
        try:
            # Google AI pricing page structure
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) < 2:
                    continue
                
                headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(['th', 'td'])]
                
                model_col = None
                input_col = None
                output_col = None
                context_col = None
                
                for i, header in enumerate(headers):
                    if 'model' in header or 'gemini' in header:
                        model_col = i
                    elif 'input' in header:
                        input_col = i
                    elif 'output' in header:
                        output_col = i
                    elif 'context' in header:
                        context_col = i
                
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:
                        continue
                    
                    model_name = None
                    if model_col is not None and model_col < len(cells):
                        model_name = cells[model_col].get_text(strip=True)
                    elif len(cells) > 0:
                        model_name = cells[0].get_text(strip=True)
                    
                    if not model_name:
                        continue
                    
                    input_price = None
                    output_price = None
                    context_window = None
                    
                    if input_col is not None and input_col < len(cells):
                        input_text = cells[input_col].get_text(strip=True)
                        input_price = self._extract_price(input_text)
                    
                    if output_col is not None and output_col < len(cells):
                        output_text = cells[output_col].get_text(strip=True)
                        output_price = self._extract_price(output_text)
                    
                    if context_col is not None and context_col < len(cells):
                        context_text = cells[context_col].get_text(strip=True)
                        context_window = self._extract_context_window(context_text)
                    
                    if input_price is not None or output_price is not None:
                        pricing_data.append({
                            'provider': 'Google',
                            'model_name': model_name,
                            'input_price_per_1k': input_price,
                            'output_price_per_1k': output_price,
                            'context_window': context_window,
                            'scraped_at': timestamp
                        })
            
            # Fallback: Look for Gemini model mentions
            if not pricing_data:
                page_text = soup.get_text()
                gemini_models = re.findall(r'(Gemini\s+[0-9A-Za-z\s]+)', page_text)
                for model in set(gemini_models):
                    pricing_data.append({
                        'provider': 'Google',
                        'model_name': model.strip(),
                        'input_price_per_1k': None,
                        'output_price_per_1k': None,
                        'context_window': None,
                        'scraped_at': timestamp,
                        'note': 'Price extraction incomplete - manual verification needed'
                    })
            
            logger.info(f"Scraped {len(pricing_data)} Google AI models")
            return pricing_data
            
        except Exception as e:
            logger.error(f"Error parsing Google AI pricing: {e}", exc_info=True)
            return []

    def scrape_all_providers(self) -> List[Dict[str, Any]]:
        """
        Scrape pricing from all supported providers.

        Returns:
            List of dictionaries containing pricing data from all providers
        """
        logger.info("Scraping pricing from all providers...")
        all_pricing = []
        
        # Scrape OpenAI
        self._rate_limit_delay()
        openai_pricing = self.scrape_openai_pricing()
        all_pricing.extend(openai_pricing)
        
        # Scrape Anthropic
        self._rate_limit_delay()
        anthropic_pricing = self.scrape_anthropic_pricing()
        all_pricing.extend(anthropic_pricing)
        
        # Scrape Google
        self._rate_limit_delay()
        google_pricing = self.scrape_google_pricing()
        all_pricing.extend(google_pricing)
        
        logger.info(f"Total pricing data scraped: {len(all_pricing)} models")
        return all_pricing

    def normalize_pricing_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize pricing data to a standard format.

        Args:
            raw_data: Raw pricing data from a provider

        Returns:
            Normalized pricing data dictionary
        """
        normalized = {
            'provider': raw_data.get('provider', 'Unknown'),
            'model_name': raw_data.get('model_name', ''),
            'input_price_per_1k': raw_data.get('input_price_per_1k'),
            'output_price_per_1k': raw_data.get('output_price_per_1k'),
            'context_window': raw_data.get('context_window'),
            'scraped_at': raw_data.get('scraped_at', datetime.now().isoformat())
        }
        
        # Ensure prices are floats
        if normalized['input_price_per_1k'] is not None:
            try:
                normalized['input_price_per_1k'] = float(normalized['input_price_per_1k'])
            except (ValueError, TypeError):
                normalized['input_price_per_1k'] = None
        
        if normalized['output_price_per_1k'] is not None:
            try:
                normalized['output_price_per_1k'] = float(normalized['output_price_per_1k'])
            except (ValueError, TypeError):
                normalized['output_price_per_1k'] = None
        
        return normalized


if __name__ == "__main__":
    """Test the pricing scraper."""
    print("=" * 60)
    print("Pricing Scraper Test")
    print("=" * 60)
    print()
    
    scraper = PricingScraper(
        delay_min=2.0,
        delay_max=3.0,
        max_retries=3
    )
    
    try:
        # Test scraping all providers
        print("[Test] Scraping pricing from all providers...")
        all_pricing = scraper.scrape_all_providers()
        
        print(f"\n✅ Scraped {len(all_pricing)} pricing records")
        
        if all_pricing:
            print("\nSample pricing data:")
            for i, pricing in enumerate(all_pricing[:5], 1):
                print(f"\n  {i}. {pricing.get('provider')} - {pricing.get('model_name')}")
                print(f"     Input: ${pricing.get('input_price_per_1k', 'N/A')} per 1K tokens")
                print(f"     Output: ${pricing.get('output_price_per_1k', 'N/A')} per 1K tokens")
                if pricing.get('context_window'):
                    print(f"     Context Window: {pricing.get('context_window'):,} tokens")
        
        print("\n" + "=" * 60)
        print("✅ Test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
