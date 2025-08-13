"""
Citation management utilities for EEG Lab.

This module provides functions for managing research citations, importing from
various sources, and generating bibliographies in different formats.
"""

import re
import json
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sqlite3
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter


class CitationManager:
    """Manages research citations and bibliographies."""
    
    def __init__(self, db_path: str = 'eeg_lab.db'):
        self.db_path = db_path
        self.categories = {
            'eeg_analysis': 'EEG Analysis',
            'machine_learning': 'Machine Learning',
            'self_supervised': 'Self-Supervised Learning',
            'signal_processing': 'Signal Processing',
            'neurological_disorders': 'Neurological Disorders',
            'foundation_models': 'Foundation Models',
            'datasets': 'Datasets',
            'methodology': 'Methodology'
        }
    
    def fetch_from_doi(self, doi: str) -> Optional[Dict]:
        """
        Fetch citation data from DOI using CrossRef API.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Dictionary with citation data or None if not found
        """
        try:
            url = f"https://api.crossref.org/works/{doi}"
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'EEG-Lab/1.0 (mailto:your-email@domain.com)'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()['message']
                return self._parse_crossref_data(data)
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching DOI {doi}: {e}")
            return None
    
    def _parse_crossref_data(self, data: Dict) -> Dict:
        """Parse CrossRef API response into citation format."""
        # Extract title
        title = data.get('title', [''])[0] if data.get('title') else ''
        
        # Extract authors
        authors = []
        if 'author' in data:
            for author in data['author']:
                given = author.get('given', '')
                family = author.get('family', '')
                if family and given:
                    authors.append(f"{family}, {given}")
                elif family:
                    authors.append(family)
                elif given:
                    authors.append(given)
        
        authors_str = '; '.join(authors) if authors else ''
        
        # Extract publication year
        year = None
        if 'published-print' in data:
            year = data['published-print']['date-parts'][0][0]
        elif 'published-online' in data:
            year = data['published-online']['date-parts'][0][0]
        elif 'created' in data:
            year = data['created']['date-parts'][0][0]
        
        # Extract journal/venue
        journal = ''
        if 'container-title' in data and data['container-title']:
            journal = data['container-title'][0]
        elif 'publisher' in data:
            journal = data['publisher']
        
        # Extract other metadata
        abstract = data.get('abstract', '')
        url = data.get('URL', '')
        citations_count = data.get('is-referenced-by-count', 0)
        
        # Try to categorize based on subject
        category = self._auto_categorize(title, abstract, journal)
        
        return {
            'title': title,
            'authors': authors_str,
            'year': year,
            'journal': journal,
            'abstract': abstract,
            'url': url,
            'citations_count': citations_count,
            'category': category
        }
    
    def _auto_categorize(self, title: str, abstract: str, journal: str) -> str:
        """Automatically categorize citation based on content."""
        text = f"{title} {abstract} {journal}".lower()
        
        # Define keywords for each category
        keywords = {
            'eeg_analysis': ['eeg', 'electroencephalography', 'brain waves', 'neural oscillations'],
            'machine_learning': ['machine learning', 'deep learning', 'neural network', 'classification'],
            'self_supervised': ['self-supervised', 'unsupervised', 'contrastive learning', 'representation learning'],
            'signal_processing': ['signal processing', 'filtering', 'frequency analysis', 'spectral'],
            'neurological_disorders': ['epilepsy', 'seizure', 'alzheimer', 'parkinson', 'stroke', 'disorder'],
            'foundation_models': ['foundation model', 'transformer', 'bert', 'gpt', 'pretrained'],
            'datasets': ['dataset', 'database', 'corpus', 'benchmark'],
            'methodology': ['methodology', 'framework', 'approach', 'technique']
        }
        
        # Score each category
        scores = {}
        for category, category_keywords in keywords.items():
            score = sum(1 for keyword in category_keywords if keyword in text)
            if score > 0:
                scores[category] = score
        
        # Return category with highest score, default to 'methodology'
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'methodology'
    
    def import_from_bibtex(self, bibtex_content: str) -> List[Dict]:
        """
        Import citations from BibTeX content.
        
        Args:
            bibtex_content: BibTeX formatted string
            
        Returns:
            List of citation dictionaries
        """
        try:
            parser = BibTexParser()
            bib_database = bibtexparser.loads(bibtex_content, parser=parser)
            
            citations = []
            for entry in bib_database.entries:
                citation = {
                    'title': entry.get('title', ''),
                    'authors': entry.get('author', ''),
                    'year': int(entry.get('year', 0)) if entry.get('year', '').isdigit() else None,
                    'journal': entry.get('journal', '') or entry.get('booktitle', ''),
                    'doi': entry.get('doi', ''),
                    'url': entry.get('url', ''),
                    'abstract': entry.get('abstract', ''),
                    'category': self._auto_categorize(
                        entry.get('title', ''),
                        entry.get('abstract', ''),
                        entry.get('journal', '')
                    )
                }
                citations.append(citation)
            
            return citations
            
        except Exception as e:
            print(f"Error parsing BibTeX: {e}")
            return []
    
    def export_to_bibtex(self, citation_ids: Optional[List[int]] = None) -> str:
        """
        Export citations to BibTeX format.
        
        Args:
            citation_ids: List of citation IDs to export. If None, exports all.
            
        Returns:
            BibTeX formatted string
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if citation_ids:
            placeholders = ','.join('?' * len(citation_ids))
            cursor.execute(f'SELECT * FROM citations WHERE id IN ({placeholders})', citation_ids)
        else:
            cursor.execute('SELECT * FROM citations ORDER BY year DESC, authors ASC')
        
        citations = cursor.fetchall()
        conn.close()
        
        bibtex_entries = []
        for citation in citations:
            # Create citation key
            first_author = citation[2].split(',')[0].strip() if citation[2] else 'Unknown'
            key = f"{first_author.replace(' ', '').lower()}{citation[3]}"
            
            entry = f"@article{{{key},\n"
            entry += f"  title={{{citation[1]}}},\n"
            entry += f"  author={{{citation[2]}}},\n"
            entry += f"  year={{{citation[3]}}},\n"
            
            if citation[4]:  # journal
                entry += f"  journal={{{citation[4]}}},\n"
            if citation[6]:  # doi
                entry += f"  doi={{{citation[6]}}},\n"
            if citation[7]:  # url
                entry += f"  url={{{citation[7]}}},\n"
            if citation[8]:  # abstract
                entry += f"  abstract={{{citation[8]}}},\n"
            
            entry = entry.rstrip(',\n') + '\n}'
            bibtex_entries.append(entry)
        
        return '\n\n'.join(bibtex_entries)
    
    def export_to_apa(self, citation_ids: Optional[List[int]] = None) -> str:
        """Export citations in APA format."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if citation_ids:
            placeholders = ','.join('?' * len(citation_ids))
            cursor.execute(f'SELECT * FROM citations WHERE id IN ({placeholders})', citation_ids)
        else:
            cursor.execute('SELECT * FROM citations ORDER BY year DESC, authors ASC')
        
        citations = cursor.fetchall()
        conn.close()
        
        apa_citations = []
        for citation in citations:
            # Format: Authors (Year). Title. Journal. DOI
            apa = f"{citation[2]} ({citation[3]}). {citation[1]}."
            
            if citation[4]:  # journal
                apa += f" {citation[4]}."
            
            if citation[6]:  # doi
                apa += f" https://doi.org/{citation[6]}"
            elif citation[7]:  # url
                apa += f" Retrieved from {citation[7]}"
            
            apa_citations.append(apa)
        
        return '\n\n'.join(apa_citations)
    
    def search_citations(self, query: str, category: Optional[str] = None, 
                        year_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Search citations by query, category, and year range.
        
        Args:
            query: Search query
            category: Filter by category
            year_range: Tuple of (min_year, max_year)
            
        Returns:
            List of matching citations
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = """
            SELECT * FROM citations 
            WHERE (title LIKE ? OR authors LIKE ? OR abstract LIKE ? OR notes LIKE ?)
        """
        params = [f'%{query}%'] * 4
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        if year_range:
            sql += " AND year BETWEEN ? AND ?"
            params.extend(year_range)
        
        sql += " ORDER BY relevance_score DESC, year DESC"
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        return [self._citation_tuple_to_dict(citation) for citation in results]
    
    def _citation_tuple_to_dict(self, citation_tuple) -> Dict:
        """Convert citation tuple from database to dictionary."""
        return {
            'id': citation_tuple[0],
            'title': citation_tuple[1],
            'authors': citation_tuple[2],
            'year': citation_tuple[3],
            'journal': citation_tuple[4],
            'category': citation_tuple[5],
            'doi': citation_tuple[6],
            'url': citation_tuple[7],
            'abstract': citation_tuple[8],
            'notes': citation_tuple[9],
            'tags': citation_tuple[10],
            'relevance_score': citation_tuple[11],
            'citations_count': citation_tuple[12],
            'bookmarked': citation_tuple[13],
            'date_added': citation_tuple[14]
        }
    
    def generate_reading_list(self, category: Optional[str] = None, 
                            min_relevance: int = 3) -> List[Dict]:
        """
        Generate a reading list of highly relevant papers.
        
        Args:
            category: Filter by category
            min_relevance: Minimum relevance score (1-5)
            
        Returns:
            List of citations sorted by relevance and citation count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = "SELECT * FROM citations WHERE relevance_score >= ?"
        params = [min_relevance]
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        sql += " ORDER BY relevance_score DESC, citations_count DESC, year DESC"
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        return [self._citation_tuple_to_dict(citation) for citation in results]
    
    def get_citation_statistics(self) -> Dict:
        """Get statistics about the citation database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total citations
        cursor.execute('SELECT COUNT(*) FROM citations')
        total = cursor.fetchone()[0]
        
        # By category
        cursor.execute('SELECT category, COUNT(*) FROM citations GROUP BY category')
        by_category = dict(cursor.fetchall())
        
        # By year
        cursor.execute('SELECT year, COUNT(*) FROM citations GROUP BY year ORDER BY year DESC')
        by_year = dict(cursor.fetchall())
        
        # Bookmarked
        cursor.execute('SELECT COUNT(*) FROM citations WHERE bookmarked = 1')
        bookmarked = cursor.fetchone()[0]
        
        # Average relevance
        cursor.execute('SELECT AVG(relevance_score) FROM citations WHERE relevance_score IS NOT NULL')
        avg_relevance = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_citations': total,
            'by_category': by_category,
            'by_year': by_year,
            'bookmarked_count': bookmarked,
            'average_relevance': round(avg_relevance, 2)
        }


# Utility functions
def extract_doi_from_url(url: str) -> Optional[str]:
    """Extract DOI from various URL formats."""
    doi_patterns = [
        r'doi\.org/(10\.\d+/.+)',
        r'doi:(10\.\d+/.+)',
        r'DOI:(10\.\d+/.+)',
        r'(10\.\d+/.+)'
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def validate_citation_data(citation: Dict) -> Tuple[bool, List[str]]:
    """
    Validate citation data for completeness and correctness.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    if not citation.get('title'):
        errors.append('Title is required')
    
    if not citation.get('authors'):
        errors.append('Authors are required')
    
    if not citation.get('year'):
        errors.append('Year is required')
    elif not isinstance(citation['year'], int) or citation['year'] < 1800 or citation['year'] > 2030:
        errors.append('Year must be a valid integer between 1800 and 2030')
    
    if not citation.get('category'):
        errors.append('Category is required')
    
    # Optional field validation
    if citation.get('doi') and not re.match(r'^10\.\d+/.+', citation['doi']):
        errors.append('DOI format is invalid')
    
    if citation.get('url') and not re.match(r'^https?://.+', citation['url']):
        errors.append('URL format is invalid')
    
    if citation.get('relevance_score') and not (1 <= citation['relevance_score'] <= 5):
        errors.append('Relevance score must be between 1 and 5')
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # Example usage
    manager = CitationManager()
    
    # Fetch citation from DOI
    doi = "10.1038/nature14539"
    citation_data = manager.fetch_from_doi(doi)
    if citation_data:
        print("Fetched citation:")
        print(f"Title: {citation_data['title']}")
        print(f"Authors: {citation_data['authors']}")
        print(f"Year: {citation_data['year']}")
        print(f"Category: {citation_data['category']}")
    
    # Get statistics
    stats = manager.get_citation_statistics()
    print(f"\nCitation Statistics:")
    print(f"Total citations: {stats['total_citations']}")
    print(f"Average relevance: {stats['average_relevance']}")
