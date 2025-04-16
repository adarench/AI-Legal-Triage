import re
import docx
from typing import List, Dict

class ClauseParser:
    def __init__(self):
        pass
        
    def parse_file(self, file_path: str) -> List[str]:
        """Parse a text or docx file and extract clauses."""
        if file_path.endswith('.docx'):
            return self._parse_docx(file_path)
        elif file_path.endswith('.txt'):
            return self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _parse_txt(self, file_path: str) -> List[str]:
        """Parse a text file and split into logical clauses."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self.extract_clauses(text)
    
    def _parse_docx(self, file_path: str) -> List[str]:
        """Parse a docx file and split into logical clauses."""
        doc = docx.Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs])
        return self.extract_clauses(full_text)
    
    def extract_clauses(self, text: str) -> List[str]:
        """Extract logical clauses from text using various heuristics."""
        # Remove multiple newlines and whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split by common clause separators
        # This is a simplistic approach - could be improved with NLP techniques
        clause_markers = [
            r'\d+\.\d+',  # Numbered clauses like "1.1", "2.3"
            r'SECTION \d+',
            r'Section \d+',
            r'Article \d+',
            r'ARTICLE \d+',
            r';',  # Semicolons often separate clauses
        ]
        
        pattern = '|'.join(f'({marker})' for marker in clause_markers)
        
        # Split text by markers but keep the markers
        segments = re.split(f'({pattern})', text)
        
        # Combine markers with their following text
        clauses = []
        i = 0
        while i < len(segments):
            if i+1 < len(segments) and any(re.match(marker, segments[i]) for marker in clause_markers):
                clauses.append(segments[i] + segments[i+1])
                i += 2
            else:
                if segments[i].strip():
                    clauses.append(segments[i].strip())
                i += 1
                
        # Filter out empty clauses and clean them
        clauses = [clause.strip() for clause in clauses if clause.strip()]
        
        return clauses
