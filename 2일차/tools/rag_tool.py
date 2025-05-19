# AI-Service-Mini-Project-main/tools/rag_tool.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# .env 파일에서 API 키 로드
load_dotenv()
# OPENAI_API_KEY는 OpenAIEmbeddings에서 자동으로 환경변수를 참조합니다.

# 상수 정의
PDF_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Artificial General Intelligence, Intelligent Agents, Voice Intelligence.pdf')
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_vectorstore_startup_info') # 벡터 저장소 저장 경로

class RAGTool:
    def __init__(self, pdf_path: str = PDF_FILE_PATH, vectorstore_path: str = VECTORSTORE_PATH):
        self.pdf_path = pdf_path
        self.vectorstore_path = vectorstore_path
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        """
        벡터 저장소가 이미 존재하면 로드하고, 없으면 PDF에서 새로 생성합니다.
        """
        if os.path.exists(self.vectorstore_path):
            print(f"--- 기존 벡터 저장소 로드: {self.vectorstore_path} ---")
            return FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True) # allow_dangerous_deserialization 추가
        else:
            print(f"--- PDF에서 새로운 벡터 저장소 생성: {self.pdf_path} ---")
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {self.pdf_path}")

            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            print(f"문서 청크 개수: {len(texts)}")

            vectorstore = FAISS.from_documents(texts, self.embeddings)
            vectorstore.save_local(self.vectorstore_path)
            print(f"--- 벡터 저장소 저장 완료: {self.vectorstore_path} ---")
            return vectorstore

    def search_documents(self, query: str, k: int = 3) -> list:
        """
        주어진 쿼리와 가장 유사한 k개의 문서를 벡터 저장소에서 검색합니다.
        LangChain의 Document 객체 리스트를 반환합니다.
        """
        print(f"--- RAG 문서 검색 실행 (k={k}): {query} ---")
        if self.vectorstore:
            # 유사도 검색 결과를 가져옵니다. 각 Document 객체는 page_content와 metadata를 가집니다.
            similar_docs = self.vectorstore.similarity_search(query, k=k)
            # 필요한 정보만 추출하여 반환 (예: page_content 와 출처 페이지)
            results = []
            for doc in similar_docs:
                source_page = doc.metadata.get('page', 'N/A') # PyPDFLoader는 페이지 번호를 metadata에 저장
                results.append({
                    "content": doc.page_content,
                    "source_document": os.path.basename(self.pdf_path),
                    "source_page": source_page
                })
            print(f"RAG 검색 결과 ({len(results)}개): {results}")
            return results
        return []

if __name__ == '__main__':
    # 간단한 테스트
    rag_tool_instance = RAGTool() # 인스턴스 생성 시 벡터 저장소 로드 또는 생성

    test_query_startup = "MAGO 회사의 주요 제품은 무엇인가요?"
    rag_results_startup = rag_tool_instance.search_documents(test_query_startup, k=2)
    if rag_results_startup:
        for i, result in enumerate(rag_results_startup):
            print(f"RAG 결과 {i+1}:")
            print(f"  문서: {result['source_document']}, 페이지: {result['source_page']}")
            print(f"  내용: {result['content'][:300]}...") # 내용이 길 수 있으므로 일부만 출력
            print("-" * 20)
    else:
        print("RAG 검색 결과가 없습니다.")

    test_query_tech = "음성 AI 기술을 사용하는 스타트업은 어디인가요?"
    rag_results_tech = rag_tool_instance.search_documents(test_query_tech, k=3)
    if rag_results_tech:
         for i, result in enumerate(rag_results_tech):
            print(f"RAG 결과 {i+1}:")
            print(f"  문서: {result['source_document']}, 페이지: {result['source_page']}")
            print(f"  내용: {result['content'][:300]}...")
            print("-" * 20)
    else:
        print("RAG 검색 결과가 없습니다.")