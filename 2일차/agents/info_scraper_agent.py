# AI-Service-Mini-Project-main/agents/info_scraper_agent.py
from typing import List, Dict, Any
import sys
import os
from states.ai_mini_design_20_3반_배민하_state_py import AdvancedStartupEvaluationState, SourceInfo # AdvancedStartupEvaluationState 임포트 확인

# 프로젝트의 루트 디렉토리를 sys.path에 추가하여 다른 모듈을 찾을 수 있도록 함
# 현재 파일(info_scraper_agent.py)의 디렉토리에서 두 단계 위로 올라가면 프로젝트 루트
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

from tools.web_search_tool import search_web_for_startup
from tools.rag_tool import RAGTool # RAGTool도 사용한다면 import
from states.ai_mini_design_20_3반_배민하_state_py import SourceInfo # SourceInfo 타입 정의 가져오기

# ../tools/web_search_tool.py 에서 search_web_for_startup 함수를 가져옵니다.
# 경로 문제를 피하기 위해 sys.path.append를 사용하거나, 프로젝트 구조를 Python 경로에 맞게 설정합니다.
# 여기서는 간단하게 상대 경로 import를 시도합니다. (실행 환경에 따라 조정 필요)
try:
    from ..tools.web_search_tool import search_web_for_startup
    from ..tools.rag_tool import RAGTool # RAGTool도 사용한다면 import
    from ..states.ai_mini_design_20_3반_배민하_state_py import SourceInfo # SourceInfo 타입 정의 가져오기
except ImportError:
    # VS Code에서 직접 실행 시 (또는 경로 문제가 있을 시) 임시로 sys.path 설정
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from tools.web_search_tool import search_web_for_startup
    from tools.rag_tool import RAGTool
    from states.ai_mini_design_20_3반_배민하_state_py import SourceInfo

# (선택사항) RAG 도구 인스턴스화 (필요한 경우)
# rag_tool_instance = RAGTool()

def run_info_scraper(state: AdvancedStartupEvaluationState) -> AdvancedStartupEvaluationState:
    print(f"DEBUG run_info_scraper entry: current_idx = {state.get('current_startup_index')}, current_name = '{state.get('current_startup_name')}'")
    all_scraped_data = []

    print("--- InfoScraperAgent 실행 시작 ---")
    startup_names = state.get("startup_names_to_compare", [])

    if not startup_names:
        print("스크랩할 스타트업 이름이 없습니다.")
        state["error_log"] = state.get("error_log", []) + ["InfoScraperAgent: 스크랩할 스타트업 이름 없음"]
        state["initial_scraped_data_all_startups"] = all_scraped_data # 빈 리스트라도 키 설정
        # if 블록 내의 DEBUG exit (오류 경로)
        print(f"DEBUG run_info_scraper exit (no startups): current_idx = {state.get('current_startup_index')}, current_name = '{state.get('current_startup_name')}', initial_scraped_data populated: {bool(state.get('initial_scraped_data_all_startups'))}")
        print(f"--- InfoScraperAgent 실행 완료. 총 {len(all_scraped_data)}개 스타트업 정보 수집 ---")
        return state

    # for 루프를 통해 모든 스타트업 정보 수집
    for name in startup_names:
        print(f"\n스타트업 '{name}' 정보 수집 중...")
        startup_data = {
            "startup_name": name,
            "scraped_sources": [],
            "raw_texts": []
        }

        # 1. 웹 검색을 통한 정보 수집
        # 검색 쿼리는 스타트업의 특성에 맞게 다양하게 구성할 수 있습니다.
        # 예: "[스타트업 이름] 회사 정보", "[스타트업 이름] 기술", "[스타트업 이름] 투자 유치" 등
        queries = [
            f"{name} 회사 소개",
            f"{name} 기술 스택",
            f"{name} 최신 뉴스",
            f"{name} 투자 정보"
        ]
        web_search_results_for_startup = []
        for query in queries:
            # search_web_for_startup 함수는 [{'source_url_or_doc': '...', 'retrieved_content_snippet': '...'}] 형태를 반환하도록 수정되었음
            results = search_web_for_startup(query)
            web_search_results_for_startup.extend(results)

        # 중복 URL 제거 (선택적)
        unique_results = []
        seen_urls = set()
        for res in web_search_results_for_startup:
            if res["source_url_or_doc"] not in seen_urls:
                unique_results.append(res)
                seen_urls.add(res["source_url_or_doc"])

        for res in unique_results:
            source_info_entry: SourceInfo = { # 타입 명시
                "source_url_or_doc": res["source_url_or_doc"],
                "retrieved_content_snippet": res["retrieved_content_snippet"]
                # "ai_confidence_score": None # 필요시 추가
            }
            startup_data["scraped_sources"].append(source_info_entry)
            startup_data["raw_texts"].append(f"출처 ({res['source_url_or_doc']}): {res['retrieved_content_snippet']}")


        # 2. (선택사항) RAG를 통한 PDF 문서 정보 보강
        # rag_query = f"{name}에 대한 정보"
        # pdf_results = rag_tool_instance.search_documents(rag_query, k=2) # 예시로 2개 문서 검색
        # for res in pdf_results:
        #     source_info_entry: SourceInfo = {
        #         "source_url_or_doc": f"{res['source_document']} (Page: {res['source_page']})",
        #         "retrieved_content_snippet": res["content"]
        #     }
        #     startup_data["scraped_sources"].append(source_info_entry)
        #     startup_data["raw_texts"].append(f"문서 출처 ({source_info_entry['source_url_or_doc']}): {res['content']}")

        # 간단한 회사 개요 생성 (LLM 사용 또는 규칙 기반) - 여기서는 수집된 정보 요약으로 대체하거나 다음 에이전트로 넘김
        # 현재 상태 정의에서는 `initial_scraped_data_all_startups`가 List[Dict[str, Any]] 이므로,
        # 여기에 저장할 데이터 형식을 `SingleStartupDetailedAnalysis`의 `company_overview`와 `scraped_urls`등에 맞춰 준비
        # 또는 Supervisor가 이 결과를 받아 `SingleStartupDetailedAnalysis`를 구성할 수도 있습니다.
        # 여기서는 `initial_scraped_data_all_startups`에 각 스타트업별 수집 정보를 담는 것으로 합니다.
        # `state.py`의 `initial_scraped_data_all_startups`는 `List[Dict[str, Any]]`로 되어 있음.
        # 이 딕셔너리의 구조를 어떻게 할지 고민 필요.
        # 예시: {'startup_name': 'MAGO', 'raw_info_sources': List[SourceInfo], 'concatenated_raw_text': str}

        # 현재 설계에서는 `initial_scraped_data_all_startups`가 List[Dict[str, Any]]로 되어있고,
        # 예시로는 `[{'startup_name': 'MAGO', 'raw_info': {...}}]`로 되어 있습니다.
        # `raw_info`의 구조를 여기서 정의합니다.
        # 여기서는 바로 `state.individual_detailed_analyses`의 일부를 채우는 방향으로 진행해볼 수도 있고,
        # Supervisor가 후처리하도록 원시 데이터를 넘길 수도 있습니다.
        # `README.md`의 아키텍처 설명을 보면, `InfoScraperAgent`의 결과가 `initial_scraped_data_all_startups`에 저장되고,
        # 이를 Supervisor가 받아서 전문 분석 에이전트들에게 전달하는 흐름입니다.
        # 따라서, `initial_scraped_data_all_startups`에는 각 스타트업별로 스크랩된 데이터를 잘 정리해서 넣어야 합니다.

        # `raw_texts`를 하나의 문자열로 합치거나, LLM이 요약하도록 할 수 있습니다.
        # 여기서는 `scraped_sources`를 주요 결과로 저장.
        current_startup_scraped_data = {
            "startup_name": name,
            "retrieved_sources": startup_data["scraped_sources"]
        }
        all_scraped_data.append(current_startup_scraped_data)
        print(f"스타트업 '{name}' 정보 수집 완료. 수집된 출처 개수: {len(startup_data['scraped_sources'])}")

    state["initial_scraped_data_all_startups"] = all_scraped_data

    # VVVV --- 이 DEBUG 프린트문이 중요합니다 (성공 경로) --- VVVV
    print(f"DEBUG run_info_scraper exit (success): current_idx = {state.get('current_startup_index')}, current_name = '{state.get('current_startup_name')}', initial_scraped_data populated: {bool(state.get('initial_scraped_data_all_startups'))}")
    # ^^^^ --- 여기까지 --- ^^^^

    print(f"--- InfoScraperAgent 실행 완료. 총 {len(all_scraped_data)}개 스타트업 정보 수집 ---")
    return state

if __name__ == '__main__':
    # InfoScraperAgent 직접 실행 테스트
    # `states.ai_mini_design_20_3반_배민하_state_py` 에 정의된 AdvancedStartupEvaluationState 구조를 따르는
    # 테스트용 state 딕셔너리를 만듭니다.
    # 실제 LangGraph 실행 시에는 Supervisor가 이 state를 초기화하고 관리합니다.
    test_state = {
        "startup_names_to_compare": ["MAGO", "The Plan G"], # 테스트할 스타트업 이름
        "user_defined_criteria_weights": None,
        "initial_scraped_data_all_startups": [],
        "individual_detailed_analyses": [],
        "comparative_analysis_output": {},
        "final_comparison_report_text": "",
        "messages": [],
        "error_log": []
    }
    updated_state = run_info_scraper(test_state)

    print("\n--- 최종 State 업데이트 결과 (initial_scraped_data_all_startups 일부) ---")
    if updated_state.get("initial_scraped_data_all_startups"):
        for data in updated_state["initial_scraped_data_all_startups"]:
            print(f"\nStartup: {data['startup_name']}")
            print(f"  Retrieved Sources Count: {len(data['retrieved_sources'])}")
            if data['retrieved_sources']:
                print(f"  Example Source URL: {data['retrieved_sources'][0]['source_url_or_doc']}")
                # print(f"  Example Snippet: {data['retrieved_sources'][0]['retrieved_content_snippet'][:100]}...")
    else:
        print("수집된 데이터가 없습니다.")

    if updated_state.get("error_log"):
        print("\n--- 오류 로그 ---")
        for error in updated_state["error_log"]:
            print(error)