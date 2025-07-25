# src/forum_scraper.py

"""
Crypto forum and discussion board scraper for additional sentiment data.
Targets: Bitcointalk.org, Stack Overflow, GitHub issues, Medium articles
"""
import os
import time
import logging
import hashlib
import requests
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from database import CryptoPulseDB

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forum_scraper.log'),
        logging.StreamHandler()
    ]
)

class CryptoForumScraper:
    def __init__(self):
        self.db = CryptoPulseDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Forum sources configuration
        self.forum_sources = {
            'bitcointalk': {
                'base_url': 'https://bitcointalk.org',
                'ethereum_board': '/index.php?board=160.0',  # Ethereum board
                'search_url': '/index.php?action=search2',
                'search_terms': ['ethereum', 'ETH', 'smart contract', 'vitalik', 'EIP']
            },
            'stackoverflow': {
                'base_url': 'https://api.stackexchange.com/2.3',
                'search_endpoint': '/search/advanced',
                'tags': ['ethereum', 'solidity', 'smart-contracts', 'web3', 'defi'],
                'api_key': None  # Can work without API key with rate limits
            },
            'github_discussions': {
                'repositories': [
                    'ethereum/ethereum',
                    'ethereum/EIPs', 
                    'ethereum/solidity',
                    'ethereum/go-ethereum'
                ],
                'api_base': 'https://api.github.com'
            },
            'medium_crypto': {
                'base_url': 'https://medium.com',
                'search_url': '/search',
                'crypto_publications': [
                    '@ethereum',
                    '@consensys', 
                    '@mycrypto',
                    '@metamask'
                ]
            }
        }
    
    def scrape_bitcointalk_ethereum(self, max_topics=100):
        """Scrape Ethereum discussions from Bitcointalk forum."""
        logging.info("Starting Bitcointalk Ethereum board scraping")
        posts = []
        
        try:
            # Get main Ethereum board
            url = f"{self.forum_sources['bitcointalk']['base_url']}{self.forum_sources['bitcointalk']['ethereum_board']}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find topic links
                topic_links = soup.find_all('a', href=True)
                ethereum_topics = []
                
                for link in topic_links:
                    href = link.get('href')
                    if href and 'topic=' in href and any(term.lower() in link.get_text().lower() 
                                                        for term in ['ethereum', 'eth', 'smart', 'defi', 'vitalik']):
                        full_url = urljoin(self.forum_sources['bitcointalk']['base_url'], href)
                        ethereum_topics.append((full_url, link.get_text().strip()))
                
                # Scrape individual topics
                for topic_url, topic_title in ethereum_topics[:max_topics//2]:
                    try:
                        topic_posts = self.scrape_bitcointalk_topic(topic_url, topic_title)
                        posts.extend(topic_posts)
                        time.sleep(3)  # Respectful delay
                        
                    except Exception as e:
                        logging.debug(f"Failed to scrape topic {topic_url}: {e}")
                        continue
                        
        except Exception as e:
            logging.warning(f"Bitcointalk board scraping failed: {e}")
        
        return self.save_forum_posts(posts, 'bitcointalk')
    
    def scrape_bitcointalk_topic(self, topic_url, topic_title):
        """Scrape individual Bitcointalk topic."""
        posts = []
        
        try:
            response = self.session.get(topic_url, timeout=15)
            if response.status_code != 200:
                return posts
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find post containers
            post_divs = soup.find_all('div', class_='post')
            
            for post_div in post_divs[:10]:  # Limit posts per topic
                try:
                    # Extract post content
                    content_div = post_div.find('div', class_='inner')
                    if not content_div:
                        continue
                        
                    content = content_div.get_text(strip=True)
                    
                    # Extract date
                    date_elem = post_div.find('div', class_='date')
                    post_date = datetime.now()
                    if date_elem:
                        try:
                            date_text = date_elem.get_text().strip()
                            # Parse various date formats
                            post_date = datetime.strptime(date_text, '%B %d, %Y, %I:%M:%S %p')
                        except:
                            pass
                    
                    # Filter for crypto relevance
                    text = (topic_title + " " + content).lower()
                    crypto_keywords = ['ethereum', 'eth', 'crypto', 'blockchain', 'smart contract', 'defi']
                    if not any(keyword in text for keyword in crypto_keywords):
                        continue
                    
                    # Generate unique ID
                    post_id = hashlib.md5(f"{topic_url}_{content[:100]}".encode()).hexdigest()
                    
                    posts.append({
                        'id': post_id,
                        'source': 'bitcointalk',
                        'title': topic_title,
                        'content': content,
                        'created_at': post_date,
                        'url': topic_url
                    })
                    
                except Exception as e:
                    logging.debug(f"Failed to parse Bitcointalk post: {e}")
                    continue
                    
        except Exception as e:
            logging.debug(f"Failed to scrape Bitcointalk topic {topic_url}: {e}")
        
        return posts
    
    def scrape_stackoverflow_ethereum(self, max_questions=200):
        """Scrape Ethereum-related questions from Stack Overflow."""
        logging.info("Starting Stack Overflow Ethereum scraping")
        posts = []
        
        try:
            tags = self.forum_sources['stackoverflow']['tags']
            
            for tag in tags:
                try:
                    # Stack Exchange API call
                    params = {
                        'order': 'desc',
                        'sort': 'votes',
                        'tagged': tag,
                        'site': 'stackoverflow',
                        'pagesize': max_questions // len(tags),
                        'filter': 'withbody'  # Include question body
                    }
                    
                    url = f"{self.forum_sources['stackoverflow']['base_url']}/questions"
                    response = self.session.get(url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for question in data.get('items', []):
                            try:
                                # Extract question data
                                title = question.get('title', '')
                                body = question.get('body', '')
                                question_id = str(question.get('question_id', ''))
                                creation_date = datetime.fromtimestamp(question.get('creation_date', 0))
                                url = f"https://stackoverflow.com/questions/{question_id}"
                                
                                # Clean HTML from body
                                if body:
                                    body_soup = BeautifulSoup(body, 'html.parser')
                                    body = body_soup.get_text(strip=True)
                                
                                # Filter for crypto relevance
                                text = (title + " " + body).lower()
                                crypto_keywords = ['ethereum', 'eth', 'smart contract', 'solidity', 'web3', 'defi']
                                if not any(keyword in text for keyword in crypto_keywords):
                                    continue
                                
                                posts.append({
                                    'id': f"so_{question_id}",
                                    'source': 'stackoverflow',
                                    'title': title,
                                    'content': body[:2000],  # Limit content length
                                    'created_at': creation_date,
                                    'url': url
                                })
                                
                            except Exception as e:
                                logging.debug(f"Failed to parse SO question: {e}")
                                continue
                                
                    time.sleep(2)  # Rate limiting for Stack Overflow API
                    
                except Exception as e:
                    logging.warning(f"Stack Overflow scraping failed for tag {tag}: {e}")
                    continue
                    
        except Exception as e:
            logging.warning(f"Stack Overflow scraping failed: {e}")
        
        return self.save_forum_posts(posts, 'stackoverflow')
    
    def scrape_github_ethereum_discussions(self, max_per_repo=50):
        """Scrape GitHub discussions and issues from Ethereum repositories."""
        logging.info("Starting GitHub Ethereum discussions scraping")
        posts = []
        
        repositories = self.forum_sources['github_discussions']['repositories']
        
        for repo in repositories:
            try:
                # Get repository issues
                url = f"{self.forum_sources['github_discussions']['api_base']}/repos/{repo}/issues"
                params = {
                    'state': 'all',
                    'sort': 'updated',
                    'per_page': max_per_repo,
                    'labels': 'ethereum'  # Look for ethereum-labeled issues
                }
                
                response = self.session.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    issues = response.json()
                    
                    for issue in issues:
                        try:
                            title = issue.get('title', '')
                            body = issue.get('body', '') or ''
                            issue_id = str(issue.get('id', ''))
                            created_at = datetime.fromisoformat(issue.get('created_at', '').replace('Z', '+00:00'))
                            html_url = issue.get('html_url', '')
                            
                            # Filter for crypto relevance
                            text = (title + " " + body).lower()
                            crypto_keywords = ['ethereum', 'eth', 'smart contract', 'eip', 'gas', 'block']
                            if not any(keyword in text for keyword in crypto_keywords):
                                continue
                            
                            posts.append({
                                'id': f"gh_{issue_id}",
                                'source': f'github_{repo.replace("/", "_")}',
                                'title': title,
                                'content': body[:1500],  # Limit content length
                                'created_at': created_at.replace(tzinfo=None),
                                'url': html_url
                            })
                            
                        except Exception as e:
                            logging.debug(f"Failed to parse GitHub issue: {e}")
                            continue
                            
                time.sleep(3)  # Rate limiting for GitHub API
                
            except Exception as e:
                logging.warning(f"GitHub scraping failed for {repo}: {e}")
                continue
        
        return self.save_forum_posts(posts, 'github_discussions')
    
    def scrape_medium_crypto_articles(self, max_articles=100):
        """Scrape crypto articles from Medium publications."""
        logging.info("Starting Medium crypto articles scraping")
        posts = []
        
        publications = self.forum_sources['medium_crypto']['crypto_publications']
        
        for pub in publications:
            try:
                # Medium publication URL
                pub_url = f"{self.forum_sources['medium_crypto']['base_url']}/{pub}"
                response = self.session.get(pub_url, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find article links (Medium's structure varies)
                    article_links = soup.find_all('a', href=True)
                    ethereum_articles = []
                    
                    for link in article_links:
                        href = link.get('href')
                        text = link.get_text().lower()
                        
                        if (href and '/p/' in href and 
                            any(term in text for term in ['ethereum', 'eth', 'defi', 'smart contract'])):
                            full_url = urljoin(self.forum_sources['medium_crypto']['base_url'], href)
                            ethereum_articles.append(full_url)
                    
                    # Scrape individual articles
                    for article_url in ethereum_articles[:max_articles//len(publications)]:
                        try:
                            article = self.scrape_medium_article(article_url)
                            if article:
                                posts.append(article)
                            time.sleep(2)
                            
                        except Exception as e:
                            logging.debug(f"Failed to scrape Medium article {article_url}: {e}")
                            continue
                            
                time.sleep(5)  # Rate limiting
                
            except Exception as e:
                logging.warning(f"Medium scraping failed for {pub}: {e}")
                continue
        
        return self.save_forum_posts(posts, 'medium_crypto')
    
    def scrape_medium_article(self, article_url):
        """Scrape individual Medium article."""
        try:
            response = self.session.get(article_url, timeout=15)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Extract content paragraphs
            content_elems = soup.find_all('p')
            content = " ".join([p.get_text(strip=True) for p in content_elems[:15]])
            
            # Filter for crypto relevance
            text = (title + " " + content).lower()
            crypto_keywords = ['ethereum', 'eth', 'crypto', 'blockchain', 'defi', 'smart contract']
            if not any(keyword in text for keyword in crypto_keywords):
                return None
            
            # Generate unique ID
            article_id = hashlib.md5(article_url.encode()).hexdigest()
            
            return {
                'id': f"medium_{article_id}",
                'source': 'medium_crypto',
                'title': title,
                'content': content[:2000],
                'created_at': datetime.now(),  # Medium doesn't always expose dates easily
                'url': article_url
            }
            
        except Exception as e:
            logging.debug(f"Failed to scrape Medium article {article_url}: {e}")
            return None
    
    def save_forum_posts(self, posts, source_type):
        """Save forum posts to database (reuse news_articles table)."""
        if not posts:
            logging.info(f"No posts collected from {source_type}")
            return 0
        
        # Convert to news articles format
        articles = []
        for post in posts:
            articles.append({
                'id': post['id'],
                'source': post['source'],
                'title': post['title'],
                'content': post['content'],
                'published_at': post['created_at'],
                'url': post['url']
            })
        
        df = pd.DataFrame(articles)
        df = df.drop_duplicates(subset=['id'])
        
        # Check for existing records
        new_articles = []
        for _, article in df.iterrows():
            if not self.db.record_exists('news_articles', article['id']):
                new_articles.append(article.to_dict())
        
        if new_articles:
            new_df = pd.DataFrame(new_articles)
            inserted = self.db.insert_news_articles(new_df)
            logging.info(f"{source_type}: {len(posts)} processed → {inserted} new saved")
            return inserted
        
        logging.info(f"{source_type}: {len(posts)} processed → 0 new (all duplicates)")
        return 0
    
    def collect_all_forums(self):
        """Run all forum collection methods."""
        logging.info("=== Starting Comprehensive Forum Collection ===")
        total = 0
        
        # Stack Overflow (most reliable)
        total += self.scrape_stackoverflow_ethereum(max_questions=100)
        
        # GitHub discussions
        total += self.scrape_github_ethereum_discussions(max_per_repo=30)
        
        # Bitcointalk (can be slow/unreliable)
        # total += self.scrape_bitcointalk_ethereum(max_topics=50)
        
        # Medium articles
        # total += self.scrape_medium_crypto_articles(max_articles=50)
        
        logging.info(f"=== Forum Collection Complete: {total} new posts ===")
        return total

def main():
    """Main function for forum collection."""
    scraper = CryptoForumScraper()
    total = scraper.collect_all_forums()
    print(f"Forum collection complete. Collected {total} new posts.")

if __name__ == "__main__":
    main()