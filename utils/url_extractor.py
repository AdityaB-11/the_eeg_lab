"""
URL Metadata Extractor for Academic Papers
Extracts citation information from various academic paper URLs
"""

import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Optional


def extract_arxiv_metadata(url: str) -> Optional[Dict]:
    """Extract metadata from arXiv URLs"""
    try:
        # Convert to abs URL if needed
        if '/pdf/' in url:
            url = url.replace('/pdf/', '/abs/')
        if url.endswith('.pdf'):
            url = url.replace('.pdf', '')
        
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1', class_='title mathjax')
        title = title_elem.get_text().replace('Title:', '').strip() if title_elem else ''
        
        # Extract authors
        authors_elem = soup.find('div', class_='authors')
        authors = []
        if authors_elem:
            author_links = authors_elem.find_all('a')
            authors = [link.get_text().strip() for link in author_links]
        authors_str = ', '.join(authors)
        
        # Extract abstract
        abstract_elem = soup.find('blockquote', class_='abstract mathjax')
        abstract = ''
        if abstract_elem:
            abstract = abstract_elem.get_text().replace('Abstract:', '').strip()
        
        # Extract year from submission date
        year = None
        date_elem = soup.find('div', class_='dateline')
        if date_elem:
            date_text = date_elem.get_text()
            year_match = re.search(r'\b(20\d{2})\b', date_text)
            if year_match:
                year = int(year_match.group(1))
        
        # Extract arXiv ID for DOI-like reference
        arxiv_id = ''
        id_match = re.search(r'arxiv\.org/abs/([^/\s]+)', url)
        if id_match:
            arxiv_id = id_match.group(1)
        
        return {
            'title': title,
            'authors': authors_str,
            'year': year,
            'journal': 'arXiv preprint',
            'abstract': abstract,
            'doi': f'arXiv:{arxiv_id}' if arxiv_id else '',
            'url': url,
            'keywords': 'preprint'
        }
    except Exception as e:
        print(f"Error extracting arXiv metadata: {e}")
        return None


def extract_pubmed_metadata(url: str) -> Optional[Dict]:
    """Extract metadata from PubMed URLs"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1', class_='heading-title')
        title = title_elem.get_text().strip() if title_elem else ''
        
        # Extract authors
        authors = []
        author_elems = soup.find_all('a', class_='full-name')
        authors = [elem.get_text().strip() for elem in author_elems]
        authors_str = ', '.join(authors)
        
        # Extract journal and year
        journal = ''
        year = None
        citation_elem = soup.find('button', class_='journal-actions-trigger')
        if citation_elem:
            citation_text = citation_elem.get_text()
            # Extract year
            year_match = re.search(r'\b(20\d{2})\b', citation_text)
            if year_match:
                year = int(year_match.group(1))
            # Extract journal
            journal_match = re.search(r'^([^.]+)', citation_text)
            if journal_match:
                journal = journal_match.group(1).strip()
        
        # Extract abstract
        abstract_elem = soup.find('div', class_='abstract-content selected')
        abstract = abstract_elem.get_text().strip() if abstract_elem else ''
        
        # Extract DOI
        doi = ''
        doi_elem = soup.find('span', class_='identifier doi')
        if doi_elem:
            doi_link = doi_elem.find('a')
            if doi_link:
                doi = doi_link.get_text().strip()
        
        return {
            'title': title,
            'authors': authors_str,
            'year': year,
            'journal': journal,
            'abstract': abstract,
            'doi': doi,
            'url': url,
            'keywords': 'peer-reviewed'
        }
    except Exception as e:
        print(f"Error extracting PubMed metadata: {e}")
        return None


def extract_ieee_metadata(url: str) -> Optional[Dict]:
    """Extract metadata from IEEE Xplore URLs"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1', class_='document-title')
        title = title_elem.get_text().strip() if title_elem else ''
        
        # Extract authors
        authors = []
        author_elems = soup.find_all('span', class_='authors-info')
        for elem in author_elems:
            author_links = elem.find_all('a')
            authors.extend([link.get_text().strip() for link in author_links])
        authors_str = ', '.join(authors)
        
        # Extract publication info
        journal = ''
        year = None
        pub_elem = soup.find('div', class_='u-pb-1 stats-document-abstract-publishedIn')
        if pub_elem:
            pub_text = pub_elem.get_text()
            year_match = re.search(r'\b(20\d{2})\b', pub_text)
            if year_match:
                year = int(year_match.group(1))
            journal = pub_text.split('Published in:')[-1].strip() if 'Published in:' in pub_text else ''
        
        # Extract abstract
        abstract_elem = soup.find('div', class_='abstract-text row')
        abstract = abstract_elem.get_text().strip() if abstract_elem else ''
        
        # Extract DOI
        doi = ''
        doi_elem = soup.find('div', class_='stats-document-abstract-doi')
        if doi_elem:
            doi_text = doi_elem.get_text()
            doi_match = re.search(r'10\.\d+/[^\s]+', doi_text)
            if doi_match:
                doi = doi_match.group(0)
        
        return {
            'title': title,
            'authors': authors_str,
            'year': year,
            'journal': journal,
            'abstract': abstract,
            'doi': doi,
            'url': url,
            'keywords': 'ieee, conference'
        }
    except Exception as e:
        print(f"Error extracting IEEE metadata: {e}")
        return None


def extract_springer_metadata(url: str) -> Optional[Dict]:
    """Extract metadata from SpringerLink URLs"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1', class_='c-article-title')
        title = title_elem.get_text().strip() if title_elem else ''
        
        # Extract authors
        authors = []
        author_elems = soup.find_all('a', attrs={'data-test': 'author-name'})
        authors = [elem.get_text().strip() for elem in author_elems]
        authors_str = ', '.join(authors)
        
        # Extract journal and year
        journal = ''
        year = None
        journal_elem = soup.find('i', attrs={'data-test': 'journal-title'})
        if journal_elem:
            journal = journal_elem.get_text().strip()
        
        # Extract year from citation
        citation_elem = soup.find('div', class_='c-bibliographic-information__list')
        if citation_elem:
            citation_text = citation_elem.get_text()
            year_match = re.search(r'\b(20\d{2})\b', citation_text)
            if year_match:
                year = int(year_match.group(1))
        
        # Extract abstract
        abstract_elem = soup.find('div', class_='c-article-section__content')
        abstract = abstract_elem.get_text().strip() if abstract_elem else ''
        
        # Extract DOI
        doi = ''
        doi_elem = soup.find('span', class_='c-bibliographic-information__value')
        if doi_elem and 'doi.org' in doi_elem.get_text():
            doi_match = re.search(r'10\.\d+/[^\s]+', doi_elem.get_text())
            if doi_match:
                doi = doi_match.group(0)
        
        return {
            'title': title,
            'authors': authors_str,
            'year': year,
            'journal': journal,
            'abstract': abstract,
            'doi': doi,
            'url': url,
            'keywords': 'springer'
        }
    except Exception as e:
        print(f"Error extracting Springer metadata: {e}")
        return None


def extract_general_metadata(url: str) -> Optional[Dict]:
    """Extract metadata from general URLs using HTML meta tags"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = ''
        # Try different title sources
        title_elem = soup.find('meta', property='og:title')
        if title_elem:
            title = title_elem.get('content', '')
        else:
            title_elem = soup.find('title')
            if title_elem:
                title = title_elem.get_text().strip()
        
        # Extract description/abstract
        abstract = ''
        desc_elem = soup.find('meta', property='og:description')
        if desc_elem:
            abstract = desc_elem.get('content', '')
        else:
            desc_elem = soup.find('meta', attrs={'name': 'description'})
            if desc_elem:
                abstract = desc_elem.get('content', '')
        
        # Extract authors from meta tags
        authors = ''
        author_elem = soup.find('meta', attrs={'name': 'author'})
        if author_elem:
            authors = author_elem.get('content', '')
        else:
            # Try to find author in text
            author_patterns = [
                r'(?:Author[s]?|By):\s*([^<\n]+)',
                r'(?:Written by|Created by):\s*([^<\n]+)'
            ]
            page_text = soup.get_text()
            for pattern in author_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    authors = match.group(1).strip()
                    break
        
        # Extract year
        year = None
        # Try to find year in URL or text
        year_match = re.search(r'\b(20\d{2})\b', url + ' ' + soup.get_text()[:1000])
        if year_match:
            year = int(year_match.group(1))
        
        # Extract DOI if present
        doi = ''
        doi_match = re.search(r'10\.\d+/[^\s<>"]+', soup.get_text())
        if doi_match:
            doi = doi_match.group(0)
        
        return {
            'title': title,
            'authors': authors,
            'year': year,
            'journal': '',
            'abstract': abstract,
            'doi': doi,
            'url': url,
            'keywords': 'web'
        }
    except Exception as e:
        print(f"Error extracting general metadata: {e}")
        return None


def extract_citation_from_url(url: str) -> Optional[Dict]:
    """
    Main function to extract citation metadata from URLs
    Automatically detects the source and uses appropriate extractor
    """
    url = url.strip()
    
    # Determine the source and extract accordingly
    if 'arxiv.org' in url:
        return extract_arxiv_metadata(url)
    elif 'pubmed.ncbi.nlm.nih.gov' in url:
        return extract_pubmed_metadata(url)
    elif 'ieee.org' in url:
        return extract_ieee_metadata(url)
    elif 'springer.com' in url or 'springerlink.com' in url:
        return extract_springer_metadata(url)
    else:
        return extract_general_metadata(url)
