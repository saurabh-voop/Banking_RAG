import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin
import os
import re
from typing import List, Dict


class ImprovedBankScraper:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.data = []
        self.visited_urls = set()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.loan_urls = [
            "https://bankofmaharashtra.bank.in/personal-banking/loans/home-loan",
            "https://bankofmaharashtra.bank.in/mahabank-vehicle-loan-scheme-for-two-wheelers-loans",
            "https://bankofmaharashtra.bank.in/maha-super-flexi-housing-loan-scheme",
            "https://bankofmaharashtra.bank.in/pradhan-mantri-awas-yojana-2",
            "https://bankofmaharashtra.bank.in/personal-banking/loans/car-loan",
            "https://bankofmaharashtra.bank.in/mahabank-vehicle-loan-scheme-for-second-hand-car",
            "https://bankofmaharashtra.bank.in/topup-home-loan",
            "https://bankofmaharashtra.bank.in/educational-loans",
            "https://bankofmaharashtra.bank.in/gold-loan",
            "https://bankofmaharashtra.bank.in/personal-banking/loans/personal-loan",
            "https://bankofmaharashtra.bank.in/loan-against-property",
            "https://bankofmaharashtra.bank.in/maha-adhaar-loan",
            "https://bankofmaharashtra.bank.in/lad",
            "https://bankofmaharashtra.bank.in/mahabank-green-financing-scheme",
            "https://bankofmaharashtra.bank.in/mahabank-rooftop-solar-panel-loan"
        ]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [{**metadata, "text": text, "chunk_id": 0}]
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                **metadata,
                "text": chunk_text,
                "chunk_id": len(chunks),
                "is_chunked": True
            })
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks

    def is_valid_url(self, url):
        
        return 'bankofmaharashtra.bank.in' in url.lower()
    
    def fetch_page(self, url):
        if url in self.visited_urls:
            return None
        
        # Skip external domains
        if not self.is_valid_url(url):
            print(f"[SKIP] External URL: {url}")
            return None
            
        try:
            r = requests.get(url, headers=self.headers, timeout=15)
            if r.status_code != 200:
                print(f"[WARN] Status {r.status_code}: {url}")
                return None
            self.visited_urls.add(url)
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"[ERROR] Fetch failed {url}: {e}")
            return None

    def get_main_container(self, soup):
        return (
            soup.find("div", id="block-system-main") or
            soup.find("div", class_="block-system-main-block") or
            soup.find("div", id="region-content") or
            soup.find("div", class_="region-content") or
            soup.find("div", class_="outerWrape") or
            soup.find("main") or
            soup.body
        )

    def normalize_section(self, raw):
        raw = raw.lower()
        mapping = {
            "interest": "Interest Rate",
            "processing": "Processing Fees",
            "eligibility": "Eligibility",
            "document": "Documents Required",
            "margin": "Margin",
            "tenure": "Loan Tenure",
            "repayment": "Repayment",
            "emi": "Repayment",
            "security": "Security",
            "collateral": "Security",
            "deduction": "Deductions",
            "purpose": "Purpose",
            "feature": "Features",
            "benefit": "Features",
            "faq": "FAQ",
            "apply": "How to Apply",
            "request": "Processing Period"
        }

        for k, v in mapping.items():
            if k in raw:
                return v

        return "General"

    def scrape_linked_page(self, link_url, loan_name, section, parent_context=""):
        soup = self.fetch_page(link_url)
        if not soup:
            return

        container = self.get_main_container(soup)
        if not container:
            return

        content_parts = []
        for tag in container.find_all(["p", "li", "div"]):
            text = self.clean_text(tag.get_text(" ", strip=True))
            if text and len(text) > 20:
                content_parts.append(text)

        if content_parts:
            full_text = " ".join(content_parts)
            
            # Add parent context for better retrieval
            if parent_context:
                full_text = f"{parent_context} - Details: {full_text}"
            
            metadata = {
                "loan_name": loan_name,
                "section": section,
                "source_url": link_url,
                "content_type": "detailed_info",
                "parent_context": parent_context
            }
            
            chunks = self.chunk_text(full_text, metadata)
            self.data.extend(chunks)

    def scrape_tables(self, container, url, loan_name):
        for table in container.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue

                label = self.clean_text(cells[0].get_text(" ", strip=True))
                value_cell = cells[1]
                section = self.normalize_section(label)

                link = value_cell.find("a", href=True)
                if link:
                    link_url = urljoin(url, link["href"])
                    
                    # Only follow internal Bank of Maharashtra links
                    if not self.is_valid_url(link_url):
                        # Store reference but don't scrape
                        value = self.clean_text(value_cell.get_text(" ", strip=True))
                        if value:
                            self.data.append({
                                "loan_name": loan_name,
                                "section": section,
                                "text": f"{label}: {value}",
                                "source_url": url,
                                "content_type": "table_data",
                                "chunk_id": 0
                            })
                        continue
                    
                    parent_context = f"{loan_name} - {label}"
                    
                    # Store summary
                    self.data.append({
                        "loan_name": loan_name,
                        "section": section,
                        "text": f"{label}: See detailed information",
                        "link_url": link_url,
                        "source_url": url,
                        "content_type": "summary",
                        "chunk_id": 0
                    })

                    # Scrape detailed content
                    time.sleep(0.5)
                    self.scrape_linked_page(link_url, loan_name, section, parent_context)

                else:
                    value = self.clean_text(value_cell.get_text(" ", strip=True))
                    if value and len(value) > 10:
                        full_text = f"{label}: {value}"
                        
                        metadata = {
                            "loan_name": loan_name,
                            "section": section,
                            "source_url": url,
                            "content_type": "table_data"
                        }
                        
                        chunks = self.chunk_text(full_text, metadata)
                        self.data.extend(chunks)

    def scrape_sections(self, container, url, loan_name):
        headers = container.find_all(["h2", "h3", "h4"])

        for header in headers:
            section_title = self.clean_text(header.get_text(strip=True))
            section = self.normalize_section(section_title)
            collected_text = []

            sibling = header.find_next_sibling()

            while sibling and sibling.name not in ["h2", "h3", "h4"]:
                # Handle links
                links = sibling.find_all("a", href=True)
                for link in links:
                    link_url = urljoin(url, link["href"])
                    
                    # Only follow internal links
                    if not self.is_valid_url(link_url):
                        continue
                    
                    if link_url not in self.visited_urls:
                        link_text = self.clean_text(link.get_text(" ", strip=True) or "Related information")
                        parent_context = f"{loan_name} - {section_title}"
                        time.sleep(0.5)
                        self.scrape_linked_page(link_url, loan_name, section, parent_context)

                text = self.clean_text(sibling.get_text(" ", strip=True))
                if text and len(text) > 20:
                    collected_text.append(text)

                sibling = sibling.find_next_sibling()

            if collected_text:
                full_text = " ".join(collected_text)
                
                metadata = {
                    "loan_name": loan_name,
                    "section": section,
                    "section_title": section_title,
                    "source_url": url,
                    "content_type": "section_content"
                }
                
                chunks = self.chunk_text(full_text, metadata)
                self.data.extend(chunks)

    def scrape_page(self, url):
        soup = self.fetch_page(url)
        if not soup:
            return

        container = self.get_main_container(soup)
        if not container:
            return

        h1 = soup.find("h1")
        loan_name = self.clean_text(h1.get_text(strip=True)) if h1 else "Unknown Loan"
        
        print(f"  → Scraping: {loan_name}")

        self.scrape_tables(container, url, loan_name)
        self.scrape_sections(container, url, loan_name)

    def scrape_all(self):
        for url in self.loan_urls:
            print(f"[SCRAPING] {url}")
            self.scrape_page(url)
            time.sleep(1)

    def save(self, filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir) if os.path.dirname(base_dir) else base_dir
        filepath = os.path.join(project_root, filename)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        print(f"\n[DONE] Saved {len(self.data)} records → {filepath}")
        print(f"[INFO] Visited {len(self.visited_urls)} unique URLs")


if __name__ == "__main__":
    scraper = ImprovedBankScraper(chunk_size=400, chunk_overlap=50)
    scraper.scrape_all()
    scraper.save("data/loan_data_chunked.json")