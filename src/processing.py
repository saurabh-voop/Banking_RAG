import json
import re
from collections import Counter
from typing import List, Dict
from pathlib import Path
import sys


class DataInspector:
    """Inspect scraped data quality and apply preprocessing if needed"""
    
    def __init__(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"[INFO] Loaded {len(self.data)} records")
    
    def inspect(self):
        """Run comprehensive data quality checks"""
        print("\n" + "="*80)
        print("DATA QUALITY INSPECTION")
        print("="*80)
        
        # 1. Basic stats
        print(f"\n BASIC STATISTICS:")
        print(f"   Total records: {len(self.data)}")
        
        # 2. Check for empty texts
        empty_texts = [d for d in self.data if not d.get('text') or len(d['text'].strip()) < 10]
        print(f"   Empty/short texts: {len(empty_texts)}")
        
        # 3. Check for duplicates
        texts = [d.get('text', '') for d in self.data]
        unique_texts = len(set(texts))
        duplicates = len(texts) - unique_texts
        print(f"   Duplicate texts: {duplicates}")
        
        # 4. Text length distribution
        lengths = [len(d.get('text', '')) for d in self.data]
        print(f"   Text length - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)//len(lengths)}")
        
        # 5. Check loan coverage
        loan_names = Counter(d.get('loan_name', 'Unknown') for d in self.data)
        print(f"\n📋 LOAN COVERAGE ({len(loan_names)} unique loans):")
        for loan, count in loan_names.most_common():
            print(f"   {loan}: {count} records")
        
        # 6. Check section coverage
        sections = Counter(d.get('section', 'Unknown') for d in self.data)
        print(f"\n📑 SECTION COVERAGE ({len(sections)} unique sections):")
        for section, count in sections.most_common():
            print(f"   {section}: {count} records")
        
        # 7. Check for problematic content
        issues = self.find_issues()
        if issues:
            print(f"\n⚠️  ISSUES FOUND:")
            for issue_type, count in issues.items():
                print(f"   {issue_type}: {count}")
        else:
            print(f"\n✅ No major issues found!")
        
        # 8. Recommendation
        print(f"\n💡 RECOMMENDATION:")
        needs_preprocessing = (
            empty_texts or 
            duplicates > len(self.data) * 0.1 or 
            issues
        )
        
        if needs_preprocessing:
            print("   ⚠️  PREPROCESSING RECOMMENDED")
            print("   Run: preprocessor.preprocess_and_save()")
        else:
            print("   ✅ Data quality is good! Ready for RAG.")
        
        return needs_preprocessing
    
    def find_issues(self) -> Dict[str, int]:
        """Find specific data quality issues"""
        issues = {}
        
        for d in self.data:
            text = d.get('text', '')
            
            # Check for HTML tags
            if re.search(r'<[^>]+>', text):
                issues['HTML tags found'] = issues.get('HTML tags found', 0) + 1
            
            # Check for excessive whitespace
            if re.search(r'\s{3,}', text):
                issues['Excessive whitespace'] = issues.get('Excessive whitespace', 0) + 1
            
            # Check for special characters spam
            special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,;:!?()\-]', text)) / max(len(text), 1)
            if special_ratio > 0.1:
                issues['Too many special chars'] = issues.get('Too many special chars', 0) + 1
            
            # Check for repeated words
            words = text.lower().split()
            if len(words) > 0:
                word_counts = Counter(words)
                most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
                if most_common_count > len(words) * 0.3:
                    issues['Repeated words'] = issues.get('Repeated words', 0) + 1
        
        return issues


class DataPreprocessor:
    """Clean and optimize scraped data for RAG"""
    
    def __init__(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.cleaned_data = []
        print(f"[INIT] Loaded {len(self.raw_data)} raw records")
    
    def clean_text(self, text: str) -> str:
        """Deep clean text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '-', text)
        
        # Fix common OCR/scraping errors
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Add space after periods
        
        # Remove URLs (keep only domain if needed)
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Clean up
        text = text.strip()
        
        return text
    
    def normalize_field(self, value: str) -> str:
        """Normalize field values"""
        if not value:
            return "Unknown"
        return value.strip().title()
    
    def is_valid_record(self, record: Dict) -> bool:
        """Check if record should be kept"""
        text = record.get('text', '').strip()
        
        # Must have meaningful text
        if not text or len(text) < 20:
            return False
        
        # Must have required fields
        if not record.get('loan_name') or not record.get('section'):
            return False
        
        # Check text quality
        words = text.split()
        if len(words) < 5:
            return False
        
        # Check if it's just navigation/UI text
        ui_indicators = ['click here', 'read more', 'go to', 'back to', 'menu', 'login']
        if any(indicator in text.lower() for indicator in ui_indicators) and len(words) < 10:
            return False
        
        return True
    
    def preprocess(self) -> List[Dict]:
        """Main preprocessing pipeline"""
        print("\n[PREPROCESSING] Starting...")
        
        seen_texts = set()
        stats = {
            'total': len(self.raw_data),
            'removed_empty': 0,
            'removed_duplicate': 0,
            'removed_invalid': 0,
            'kept': 0
        }
        
        for record in self.raw_data:
            # Clean text
            cleaned_text = self.clean_text(record.get('text', ''))
            
            # Skip empty
            if not cleaned_text or len(cleaned_text) < 20:
                stats['removed_empty'] += 1
                continue
            
            # Skip duplicates
            if cleaned_text in seen_texts:
                stats['removed_duplicate'] += 1
                continue
            seen_texts.add(cleaned_text)
            
            # Create cleaned record
            cleaned_record = {
                'text': cleaned_text,
                'loan_name': self.normalize_field(record.get('loan_name')),
                'section': self.normalize_field(record.get('section')),
                'source_url': record.get('source_url', ''),
                'content_type': record.get('content_type', 'general'),
                'chunk_id': record.get('chunk_id', 0)
            }
            
            # Add optional fields
            if 'link_url' in record:
                cleaned_record['link_url'] = record['link_url']
            if 'parent_context' in record:
                cleaned_record['parent_context'] = record['parent_context']
            
            # Validate
            if not self.is_valid_record(cleaned_record):
                stats['removed_invalid'] += 1
                continue
            
            self.cleaned_data.append(cleaned_record)
            stats['kept'] += 1
        
        # Print stats
        print(f"\n[STATS] Preprocessing complete:")
        print(f"   Total records: {stats['total']}")
        print(f"   Removed (empty): {stats['removed_empty']}")
        print(f"   Removed (duplicate): {stats['removed_duplicate']}")
        print(f"   Removed (invalid): {stats['removed_invalid']}")
        print(f"   ✅ Kept: {stats['kept']}")
        print(f"   Reduction: {((stats['total'] - stats['kept']) / stats['total'] * 100):.1f}%")
        
        return self.cleaned_data
    
    def preprocess_and_save(self, output_file: str):
        """Preprocess and save cleaned data"""
        self.cleaned_data = self.preprocess()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.cleaned_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Cleaned data → {output_file}")
        return output_file


# Main execution
if __name__ == "__main__":
    import sys
    
    # Default input file
    script_dir = Path(__file__).parent
    default_file = script_dir.parent / 'data' / 'loan_data_chunked.json'
    input_file = str(default_file)
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    print("="*80)
    print("STEP 1: INSPECT DATA QUALITY")
    print("="*80)
    
    # Inspect
    inspector = DataInspector(input_file)
    needs_preprocessing = inspector.inspect()
    
    # Preprocess if needed
    if needs_preprocessing:
        print("\n" + "="*80)
        print("STEP 2: PREPROCESSING DATA")
        print("="*80)
        
        preprocessor = DataPreprocessor(input_file)
        output_file = input_file.replace('.json', '_cleaned.json')
        preprocessor.preprocess_and_save(output_file)
        
        print("\n" + "="*80)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\nUse this file for RAG: {output_file}")
    else:
        print("\n" + "="*80)
        print("✅ NO PREPROCESSING NEEDED!")
        print("="*80)
        print(f"\nUse this file for RAG: {input_file}")