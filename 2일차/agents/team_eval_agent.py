# AI-Service-Mini-Project-main/agents/team_eval_agent.py
import sys
import os
import json
from typing import Dict, List, Any, Optional

# Pydantic v1 또는 v2를 사용합니다. 프로젝트 일관성을 위해 다른 에이전트와 동일하게 설정합니다.
# 여기서는 Pydantic v2를 직접 사용하는 것으로 가정합니다.
from pydantic import BaseModel, Field

# 프로젝트 루트 디렉토리를 sys.path에 추가
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field # LangChain과 함께 Pydantic v1을 사용하려면 이 줄을 사용

from states.ai_mini_design_20_3반_배민하_state_py import TeamEvaluationOutput, SourceInfo
from tools.rag_tool import RAGTool
from tools.web_search_tool import search_web_for_startup

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# RAG 도구 인스턴스화
rag_tool_instance = RAGTool()

# LLM이 반환할 구조 정의
class LLMTeamEvaluationResponse(BaseModel):
    founder_expertise_score: Optional[float] = Field(description="창업자 전문성 점수 (1-5점 척도)")
    founder_expertise_comment: str = Field(description="창업자 전문성(관련 경험, 지식)에 대한 상세 코멘트 및 근거")
    founder_communication_score: Optional[float] = Field(description="창업자 커뮤니케이션 역량 점수 (추론, 1-5점 척도)")
    founder_communication_comment: str = Field(description="창업자의 커뮤니케이션 역량(비전 전달, 대외 활동 등)에 대한 상세 코멘트 및 근거 (추론 기반)")
    founder_execution_score: Optional[float] = Field(description="창업자 실행력 점수 (추론, 1-5점 척도)")
    founder_execution_comment: str = Field(description="창업자의 실행력(과거 성과, 프로젝트 진행 능력 등)에 대한 상세 코멘트 및 근거 (추론 기반)")
    key_team_members_summary: Optional[str] = Field(description="파악된 주요 핵심 팀원들에 대한 간략한 요약 (존재 시)")
    evaluation_confidence_score: Optional[float] = Field(description="이 팀 평가 전반에 대한 AI의 분석 신뢰도 점수 (0.0 ~ 1.0 사이).")


# 프롬프트 템플릿 정의
TEAM_EVAL_PROMPT_TEMPLATE = """
당신은 AI 스타트업의 팀 역량을 심층 분석하는 HR 전문가 및 투자 심사역입니다.
제공된 '{startup_name}' 스타트업에 대한 다음 정보를 바탕으로 팀, 특히 창업자와 주요 팀원들을 평가해주십시오.

[제공된 정보]
{information_context}

[평가 항목]
1.  **창업자 전문성**: 창업자(들)의 해당 산업 또는 기술 분야에서의 전문성, 경험, 학력 등을 분석하고 1점에서 5점 사이로 점수를 매겨주세요. 근거를 상세히 설명해주세요.
2.  **창업자 커뮤니케이션 역량 (추론)**: 창업자(들)가 비전을 명확히 전달하고, 투자자 및 팀원들과 효과적으로 소통할 수 있는 역량이 어느 정도로 보이는지 추론하여 1점에서 5점 사이로 점수를 매겨주세요. 공개된 인터뷰, 발표 내용, 회사 소개 자료 등을 바탕으로 판단해주세요.
3.  **창업자 실행력 (추론)**: 창업자(들)가 아이디어를 실제 성과로 만들어내는 실행력이 어느 정도로 보이는지 과거의 경험이나 현재까지의 프로젝트 진행 상황 등을 바탕으로 추론하여 1점에서 5점 사이로 점수를 매겨주세요.
4.  **핵심 팀원**: 창업자 외에 파악된 주요 핵심 팀원들(예: CTO, COO 등)이 있다면 간략히 요약해주시고, 그들의 역할과 중요성에 대해 언급해주세요. (정보가 부족하면 '정보 부족'으로 명시)
5.  **분석 신뢰도**: 위 팀 평가 항목들에 대한 당신의 전반적인 분석 신뢰도를 0.0에서 1.0 사이의 점수로 매겨주십시오. (예: 정보가 충분하고 명확하면 0.9, 정보가 부족하거나 추론에 많이 의존하면 0.6)

[출력 형식 지침]
다음 JSON 형식을 반드시 준수하여 답변해주십시오:
{{
    "founder_expertise_score": (float, 예: 4.5 또는 null),
    "founder_expertise_comment": "(string)",
    "founder_communication_score": (float, 예: 4.0 또는 null),
    "founder_communication_comment": "(string)",
    "founder_execution_score": (float, 예: 4.2 또는 null),
    "founder_execution_comment": "(string)",
    "key_team_members_summary": "(string, 예: 'CTO: 홍길동 - AI 개발 총괄, 10년 경력...' 또는 '정보 부족')",
    "evaluation_confidence_score": (float, 예: 0.75)
}}

스타트업 이름: {startup_name}
"""

def run_team_evaluator(startup_name: str, scraped_info: List[SourceInfo], use_rag: bool = True, use_web_search: bool = True) -> TeamEvaluationOutput:
    print(f"--- TeamEvaluationAgent 실행 시작: {startup_name} ---")

    information_snippets = []
    all_sources_for_this_evaluation: List[SourceInfo] = []

    # 1. InfoScraper가 제공한 웹 스크랩 정보 컨텍스트화
    for src_info in scraped_info:
        updated_src_info = src_info.copy()
        if 'ai_confidence_score' not in updated_src_info:
            updated_src_info['ai_confidence_score'] = None
        information_snippets.append(f"기존 수집 정보 ({updated_src_info['source_url_or_doc']}, 신뢰도: {updated_src_info.get('ai_confidence_score', 'N/A')}):\n{updated_src_info['retrieved_content_snippet']}\n---")
        all_sources_for_this_evaluation.append(updated_src_info)

    # 2. (선택적) 추가 웹 검색으로 팀/창업자 정보 보강
    if use_web_search:
        ceo_name_from_pdf = None
        # PDF에서 CEO 이름 가져오기 시도 (간단한 하드코딩 예시, 실제로는 더 정교한 로직 필요)
        # 이 정보는 RAGTool이나 다른 방법을 통해 state에서 가져오는 것이 더 좋습니다.
        if startup_name.upper() == "MAGO":
            ceo_name_from_pdf = "Hyunwoong Ko" # From PDF 
        elif startup_name.upper() == "THE PLAN G":
            ceo_name_from_pdf = "Kyunga Lee" # From PDF 

        team_queries = [
            f"{startup_name} 창업자 프로필",
            f"{startup_name} 경영진",
            f"{startup_name} 팀 소개",
        ]
        if ceo_name_from_pdf:
            team_queries.append(f"{ceo_name_from_pdf} 경력 및 인터뷰")

        for query in team_queries:
            web_results = search_web_for_startup(query)
            for res in web_results:
                source_info_entry = SourceInfo(
                    source_url_or_doc=res['source_url_or_doc'],
                    retrieved_content_snippet=res['retrieved_content_snippet'],
                    ai_confidence_score=None # 웹 검색 결과에 대한 신뢰도는 별도 평가 필요
                )
                information_snippets.append(f"웹 검색 ({res['source_url_or_doc']}, 신뢰도: N/A):\n{res['retrieved_content_snippet']}\n---")
                all_sources_for_this_evaluation.append(source_info_entry)

    # 3. (선택적) RAG를 통한 PDF 문서 정보 보강
    if use_rag:
        rag_query = f"{startup_name}의 창업자(CEO), 핵심 팀, 경영진에 대한 정보"
        pdf_results = rag_tool_instance.search_documents(rag_query, k=3)
        for res in pdf_results:
            rag_source_info: SourceInfo = {
                "source_url_or_doc": f"{res['source_document']} (Page: {res['source_page']})",
                "retrieved_content_snippet": res["content"],
                "ai_confidence_score": None # RAG 결과에 대한 신뢰도도 추후 평가 가능
            }
            information_snippets.append(f"문서 출처 ({rag_source_info['source_url_or_doc']}, 신뢰도: N/A):\n{res['content']}\n---")
            all_sources_for_this_evaluation.append(rag_source_info)


    if not information_snippets:
        print(f"{startup_name}에 대한 팀 분석 정보가 부족합니다.")
        return TeamEvaluationOutput(
            founder_expertise_score=None, founder_expertise_comment="분석 정보 부족",
            founder_communication_score=None, founder_communication_comment="분석 정보 부족",
            founder_execution_score=None, founder_execution_comment="분석 정보 부족",
            key_team_members_summary="분석 정보 부족",
            evaluation_confidence_score=0.0, # 정보 부족 시 신뢰도 0
            sources=[]
        )

    information_context = "\n".join(information_snippets)
    # print(f"LLM에 전달될 팀 컨텍스트 (일부): {information_context[:500]}...")

    prompt = ChatPromptTemplate.from_template(TEAM_EVAL_PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        response_str = chain.invoke({
            "startup_name": startup_name,
            "information_context": information_context
        }).content
        print(f"LLM 응답 (문자열): {response_str}")

        # 마크다운 코드 블록 제거 로직
        if response_str.strip().startswith("```json"):
            response_str = response_str.strip()[7:]
            if response_str.strip().endswith("```"):
                response_str = response_str.strip()[:-3]
        elif response_str.strip().startswith("```"):
             response_str = response_str.strip()[3:]
             if response_str.strip().endswith("```"):
                response_str = response_str.strip()[:-3]
        
        response_str = response_str.strip() # 추가: 앞뒤 공백 제거

        try:
            llm_response_data = json.loads(response_str)
            team_eval_output = TeamEvaluationOutput(
                founder_expertise_score=llm_response_data.get("founder_expertise_score"),
                founder_expertise_comment=llm_response_data.get("founder_expertise_comment", "파싱 오류 또는 값 없음"),
                founder_communication_score=llm_response_data.get("founder_communication_score"),
                founder_communication_comment=llm_response_data.get("founder_communication_comment", "파싱 오류 또는 값 없음"),
                founder_execution_score=llm_response_data.get("founder_execution_score"),
                founder_execution_comment=llm_response_data.get("founder_execution_comment", "파싱 오류 또는 값 없음"),
                key_team_members_summary=llm_response_data.get("key_team_members_summary"),
                evaluation_confidence_score=llm_response_data.get("evaluation_confidence_score"), # 신뢰도 점수 추가
                sources=all_sources_for_this_evaluation
            )
        except json.JSONDecodeError as e:
            print(f"LLM 응답 JSON 파싱 오류: {e}")
            print(f"오류 발생한 응답: {response_str}")
            team_eval_output = TeamEvaluationOutput(
                founder_expertise_score=None, founder_expertise_comment=f"LLM 응답 파싱 오류: {response_str}",
                founder_communication_score=None, founder_communication_comment=f"LLM 응답 파싱 오류: {response_str}",
                founder_execution_score=None, founder_execution_comment=f"LLM 응답 파싱 오류: {response_str}",
                key_team_members_summary=f"LLM 응답 파싱 오류: {response_str}",
                evaluation_confidence_score=0.1, # 파싱 오류 시 낮은 신뢰도
                sources=all_sources_for_this_evaluation
            )
    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {e}")
        team_eval_output = TeamEvaluationOutput(
            founder_expertise_score=None, founder_expertise_comment=f"LLM 호출 오류: {e}",
            founder_communication_score=None, founder_communication_comment=f"LLM 호출 오류: {e}",
            founder_execution_score=None, founder_execution_comment=f"LLM 호출 오류: {e}",
            key_team_members_summary=f"LLM 호출 오류: {e}",
            evaluation_confidence_score=0.1, # LLM 호출 오류 시 낮은 신뢰도
            sources=all_sources_for_this_evaluation
        )

    print(f"--- TeamEvaluationAgent 실행 완료: {startup_name} ---")
    print(f"분석 결과: {team_eval_output}")
    return team_eval_output

if __name__ == '__main__':
    # TeamEvaluationAgent 직접 실행 테스트
    sample_scraped_data_mago = [
        SourceInfo(source_url_or_doc="https://www.holamago.com/about", retrieved_content_snippet="MAGO의 창업자 Hyunwoong Ko는 10년간 음성 AI 분야에서 연구 개발을 진행해왔습니다. 다수의 관련 논문을 발표하고...", ai_confidence_score=0.9),
        SourceInfo(source_url_or_doc="MAGO_team_intro.com", retrieved_content_snippet="MAGO 팀은 머신러닝 전문가, UX 디자이너, 정신건강의학 자문위원으로 구성되어 있습니다.", ai_confidence_score=0.8)
    ]
    sample_scraped_data_plang = [
        SourceInfo(source_url_or_doc="https://theplang.com/team", retrieved_content_snippet="The Plan G의 대표 Kyunga Lee는 교육학 석사 출신으로, 15년간 초등 교육 콘텐츠 개발에 힘써왔습니다. 아이들의 눈높이에 맞는 학습법을 연구합니다.", ai_confidence_score=0.88),
        SourceInfo(source_url_or_doc="ThePlanG_vision.com", retrieved_content_snippet="Kyunga Lee 대표는 '모든 아이에게 즐거운 영어 학습 경험을 제공한다'는 비전을 가지고 있습니다.", ai_confidence_score=0.7)
    ]

    print("\n--- MAGO 팀 역량 평가 테스트 ---")
    mago_team_eval = run_team_evaluator(startup_name="MAGO", scraped_info=sample_scraped_data_mago, use_rag=True, use_web_search=True)
    # import json
    # print(json.dumps(mago_team_eval, indent=2, ensure_ascii=False))


    print("\n--- The Plan G 팀 역량 평가 테스트 ---")
    plang_team_eval = run_team_evaluator(startup_name="The Plan G", scraped_info=sample_scraped_data_plang, use_rag=True, use_web_search=True)
    # import json
    # print(json.dumps(plang_team_eval, indent=2, ensure_ascii=False))