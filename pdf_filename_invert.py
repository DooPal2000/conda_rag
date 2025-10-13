import os
import shutil
import unicodedata
from unidecode import unidecode
from langchain_community.document_loaders import PyPDFLoader
import re

def is_hangul(text: str) -> bool:
    """문자열에 한글이 포함되어 있는지 확인"""
    return bool(re.search(r'[\uac00-\ud7a3]', text))

def safe_pdf_loader(pdf_file: str, copy_dir: str = "data") -> PyPDFLoader:
    """
    1️⃣ 한글 파일명 감지
    2️⃣ 한글 → ASCII 변환
    3️⃣ 유니코드 정규화
    4️⃣ 복사 후 PyPDFLoader로 로딩
    
    """
    # 1️⃣ 정규화
    normalized_path = unicodedata.normalize('NFC', pdf_file)
    
    # 2️⃣ 한글 감지 및 파일명 변환
    filename = os.path.basename(normalized_path)
    if is_hangul(filename):
        ascii_name = unidecode(filename).replace(" ", "_")
        new_path = os.path.join(copy_dir, ascii_name)
        shutil.copy2(normalized_path, new_path)
        print(f"📁 한글 탐지 → 영어로 변환 후 복사: {new_path}")
    else:
        new_path = normalized_path
        print(f"📁 한글 없음 → 원본 그대로 사용: {new_path}")

    # 3️⃣ PyPDFLoader로 로딩
    loader = PyPDFLoader(new_path)
    pages = loader.load()
    print(f"📄 페이지 수: {len(pages)}")

    return loader, pages, new_path

# # 사용 예시
# pdf_file = r'data/근로기준법(법류)(제18176호)(20211119).pdf'
# loader, pages, new_path = safe_pdf_loader(pdf_file)
