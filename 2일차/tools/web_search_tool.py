# AI-Service-Mini-Project-main/tools/web_search_tool.py
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

# .env 파일에서 API 키 로드
load_dotenv()

# Tavily API 키 설정 (환경 변수 TAVILY_API_KEY를 자동으로 사용)
# os.environ["TAVILY_API_KEY"] = "tvly-your_tavily_api_key" # .env에 있다면 이 줄은 불필요

# Tavily 검색 도구 초기화
# max_results: 반환받을 검색 결과의 최대 개수
tavily_tool = TavilySearchResults(max_results=5)

def search_web_for_startup(query: str) -> list:
    """
    주어진 쿼리로 웹을 검색하여 관련 정보를 가져옵니다.
    TavilySearchResults는 기본적으로 [{ "url": "...", "content": "..." }, ...] 형태의 리스트를 반환합니다.
    """
    print(f"--- 웹 검색 실행: {query} ---")
    try:
        results = tavily_tool.invoke(query)
        # 결과가 문자열이면 (오류 메시지 등), 빈 리스트 반환 또는 적절한 처리
        if isinstance(results, str):
            print(f"웹 검색 중 오류 또는 결과 없음: {results}")
            return []
        # 결과가 비어있는 리스트일 수도 있음
        if not results:
            print("웹 검색 결과가 없습니다.")
            return []

        # content의 길이가 너무 길 경우 자르거나, 필요한 정보만 추출하는 로직 추가 가능
        # 예를 들어, 각 결과의 content를 처음 500자로 제한
        processed_results = []
        for res in results:
            processed_results.append({
                # "title": res.get("title"), # title도 필요하면 포함
                "source_url_or_doc": res.get("url"),
                "retrieved_content_snippet": res.get("content", "")[:1000] # 예: 1000자로 제한
            })
        print(f"웹 검색 결과 ({len(processed_results)}개 가공됨): {processed_results}")
        return processed_results
    
    except Exception as e:
        print(f"웹 검색 중 예외 발생: {e}")
        return []

if __name__ == '__main__':
    # 간단한 테스트
    test_query = "AI 스타트업 MAGO 기술 스택"
    search_results = search_web_for_startup(test_query)
    if search_results:
        for i, result in enumerate(search_results):
            print(f"결과 {i+1}:")
            print(f"  URL: {result.get('source_url_or_doc')}")
            print(f"  Content: {result.get('retrieved_content_snippet', '')[:200]}...") # 내용이 길 수 있으므로 일부만 출력
            print("-" * 20)
    else:
        print("검색 결과가 없습니다.")