# AI-Service-Mini-Project-main/agents/tech_eval_agent.py
import sys
import os
import json # LLM 응답이 JSON 문자열일 경우 파싱하기 위해
from typing import Dict, List, Any, Optional

# 프로젝트 루트 디렉토리를 sys.path에 추가
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field # LLM 출력을 구조화하기 위함
from pydantic import BaseModel, Field # Pydantic v2 직접 사용
# from langchain.output_parsers import PydanticOutputParser # 또는 JsonOutputParser

from states.ai_mini_design_20_3반_배민하_state_py import TechEvaluationOutput, SourceInfo
from tools.rag_tool import RAGTool # RAGTool 사용 시

# LLM 초기화 (예: GPT-4o-mini 또는 다른 모델)
# API 키는 환경변수 OPENAI_API_KEY 에서 자동으로 로드됩니다.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2) # temperature는 창의성 조절

# (선택사항) RAG 도구 인스턴스화
rag_tool_instance = RAGTool()

# LLM이 반환할 구조를 Pydantic 모델로 정의 (LangChain의 PydanticOutputParser와 함께 사용 가능)
class LLMTechEvaluationResponse(BaseModel):
    originality_score: Optional[float] = Field(description="기술 독창성 점수 (1-5점 척도)")
    originality_comment: str = Field(description="기술 독창성에 대한 상세 코멘트 및 근거")
    feasibility_score: Optional[float] = Field(description="기술 구현 가능성 점수 (1-5점 척도)")
    feasibility_comment: str = Field(description="기술 구현 가능성에 대한 상세 코멘트 및 근거")
    patent_info: Optional[str] = Field(description="관련 특허 정보 (발견 시, 요약)")
    tech_stack_summary: Optional[str] = Field(description="주요 기술 스택 요약")
    evaluation_confidence_score: Optional[float] = Field(description="이 기술 평가 전반에 대한 AI의 분석 신뢰도 점수 (0.0 ~ 1.0 사이). 제공된 정보의 질과 양을 고려하여 판단.")


# 프롬프트 템플릿 정의
TECH_EVAL_PROMPT_TEMPLATE = """
당신은 AI 스타트업의 기술 및 제품을 심층 분석하는 전문가입니다.
제공된 '{startup_name}' 스타트업에 대한 다음 정보를 바탕으로 기술적 측면을 평가해주십시오.

[제공된 정보]
{information_context}

[평가 항목]
1.  **기술 독창성**: 이 스타트업의 기술이 기존 기술과 비교하여 얼마나 독창적인지, 핵심 차별점은 무엇인지 분석하고 1점에서 5점 사이로 점수를 매겨주세요. 근거를 상세히 설명해주세요.
2.  **기술 구현 가능성**: 제시된 기술 또는 제품이 현실적으로 구현 가능성이 높은지, 기술적 난이도는 어느 정도인지 평가하고 1점에서 5점 사이로 점수를 매겨주세요. 근거를 상세히 설명해주세요.
3.  **특허 정보**: 관련 특허가 있다면 간략히 요약해주시고, 없다면 '없음'으로 명시해주세요. (정보가 부족하면 '정보 부족'으로 명시)
4.  **기술 스택**: 파악된 주요 기술 스택이 있다면 요약해주세요. (정보가 부족하면 '정보 부족'으로 명시)
5.  **분석 신뢰도**: 위 기술 평가 항목들에 대한 당신의 전반적인 분석 신뢰도를 0.0에서 1.0 사이의 점수로 매겨주십시오. (예: 정보가 충분하고 명확하면 0.9, 정보가 부족하거나 모호하면 0.5)

[출력 형식 지침]
다음 JSON 형식을 반드시 준수하여 답변해주십시오:
{{
    "originality_score": (float, 예: 4.5 또는 null),
    "originality_comment": "(string)",
    "feasibility_score": (float, 예: 4.0 또는 null),
    "feasibility_comment": "(string)",
    "patent_info": "(string, 예: '관련 특허 2건 보유: XXX, YYY ...' 또는 '없음' 또는 '정보 부족')",
    "tech_stack_summary": "(string, 예: 'Python, TensorFlow, AWS 등 활용' 또는 '정보 부족')",
    "evaluation_confidence_score": (float, 예: 0.85)
}}

스타트업 이름: {startup_name}
"""

def run_tech_evaluator(startup_name: str, scraped_info: List[SourceInfo], use_rag: bool = True) -> TechEvaluationOutput:
    """
    TechEvaluationAgent의 메인 실행 함수입니다.
    스타트업의 기술 정보를 분석하여 TechEvaluationOutput을 반환합니다.

    Args:
        startup_name (str): 분석할 스타트업의 이름.
        scraped_info (List[SourceInfo]): InfoScraperAgent가 수집한 SourceInfo 객체 리스트.
        use_rag (bool): RAGTool을 사용하여 PDF 문서 정보를 추가로 활용할지 여부.

    Returns:
        TechEvaluationOutput: 기술 평가 결과.
    """
    print(f"--- TechEvaluationAgent 실행 시작: {startup_name} ---")

    information_snippets = []
    all_sources_for_this_evaluation: List[SourceInfo] = []

    # 1. 웹 스크랩 정보 컨텍스트화
    for src_info in scraped_info:
        # AI 신뢰도 점수는 여기서 평가하기보다, LLM이 전체 정보를 보고 판단하도록 하거나,
        # 개별 정보에 대한 신뢰도를 별도로 평가하는 로직이 필요. 현재는 None으로 둠.
        # 만약 InfoScraper에서 기본 신뢰도를 설정한다면 여기서 전달 가능.
        updated_src_info = src_info.copy()
        if 'ai_confidence_score' not in updated_src_info : # 아직 신뢰도 점수가 없다면
             updated_src_info['ai_confidence_score'] = None # 또는 기본값 설정
        information_snippets.append(f"출처 ({updated_src_info['source_url_or_doc']}, 신뢰도: {updated_src_info.get('ai_confidence_score', 'N/A')}):\n{updated_src_info['retrieved_content_snippet']}\n---")
        all_sources_for_this_evaluation.append(updated_src_info)

    # 2. (선택적) RAG를 통한 PDF 문서 정보 보강
    if use_rag:
        rag_query = f"{startup_name}의 기술, 제품, 특허, 기술 스택에 대한 정보"
        pdf_results = rag_tool_instance.search_documents(rag_query, k=2) # 예시로 2개 문서 검색
        for res in pdf_results:
            rag_source_info: SourceInfo = {
                "source_url_or_doc": f"{res['source_document']} (Page: {res['source_page']})",
                "retrieved_content_snippet": res["content"],
                "ai_confidence_score": None # RAG 결과에 대한 신뢰도도 추후 평가 가능
            }
            information_snippets.append(f"문서 출처 ({rag_source_info['source_url_or_doc']}, 신뢰도: {rag_source_info.get('ai_confidence_score', 'N/A')}):\n{res['content']}\n---")
            all_sources_for_this_evaluation.append(rag_source_info)

    if not information_snippets:
        print(f"{startup_name}에 대한 분석 정보가 부족합니다.")
        # 정보 부족 시 기본값 또는 오류 처리
        return TechEvaluationOutput(
            originality_score=None, originality_comment="분석 정보 부족",
            feasibility_score=None, feasibility_comment="분석 정보 부족",
            patent_info="정보 부족", tech_stack_summary="정보 부족",
            evaluation_confidence_score=0.0, # 정보가 전혀 없으므로 신뢰도 0
            sources=[]
        )

    information_context = "\n".join(information_snippets)
    # 컨텍스트 길이 제한 고려 (필요시 요약 또는 선택적 사용)
    # print(f"LLM에 전달될 컨텍스트 (일부): {information_context[:500]}...")

    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_template(TECH_EVAL_PROMPT_TEMPLATE)
    chain = prompt | llm

    # LLM 호출
    try:
        response_str = chain.invoke({
            "startup_name": startup_name,
            "information_context": information_context
        }).content

        print(f"LLM 응답 (문자열): {response_str}")
        
        
        # <<<--- 추가된 부분 시작 --->>>
        # LLM 응답이 마크다운 코드 블록(```json ... ```)으로 감싸져 있을 경우 제거
        if response_str.strip().startswith("```json"):
            response_str = response_str.strip()[7:] # "```json\n" 제거 (json 뒤에 개행이 있을 수 있음)
            if response_str.strip().endswith("```"):
                response_str = response_str.strip()[:-3] # "\n```" 제거
        elif response_str.strip().startswith("```"): # json 명시 없이 ``` ... ``` 만 있을 경우
             response_str = response_str.strip()[3:]
             if response_str.strip().endswith("```"):
                response_str = response_str.strip()[:-3]
        # <<<--- 추가된 부분 끝 --->>>
        
        response_str = response_str.strip() # 추가: 앞뒤 공백 제거

        # LLM 응답 파싱 (JSON 형식 가정)
        # 더 견고한 파싱을 위해 LangChain의 Output Parsers (PydanticOutputParser, JsonOutputParser) 사용 권장
        try:
            llm_response_data = json.loads(response_str)
            # LLMTechEvaluationResponse 모델로 유효성 검사 (선택적이지만 권장)
            # validated_response = LLMTechEvaluationResponse(**llm_response_data)
            # tech_eval_output = TechEvaluationOutput(**validated_response.dict(), sources=all_sources_for_this_evaluation)

            # 여기서는 직접 매핑 (Pydantic 모델 없이)
            tech_eval_output = TechEvaluationOutput(
                originality_score=llm_response_data.get("originality_score"),
                originality_comment=llm_response_data.get("originality_comment", "파싱 오류 또는 값 없음"),
                feasibility_score=llm_response_data.get("feasibility_score"),
                feasibility_comment=llm_response_data.get("feasibility_comment", "파싱 오류 또는 값 없음"),
                patent_info=llm_response_data.get("patent_info"),
                tech_stack_summary=llm_response_data.get("tech_stack_summary"),
                evaluation_confidence_score=llm_response_data.get("evaluation_confidence_score"), # 신뢰도 점수 추가
                sources=all_sources_for_this_evaluation # 분석에 사용된 모든 컨텍스트 출처를 일단 포함
            )

        except json.JSONDecodeError as e:
            print(f"LLM 응답 JSON 파싱 오류: {e}")
            print(f"오류 발생한 응답: {response_str}")
            # 파싱 실패 시 기본값 또는 오류 처리
            tech_eval_output = TechEvaluationOutput(
                originality_score=None, originality_comment=f"LLM 응답 파싱 오류: {response_str}",
                feasibility_score=None, feasibility_comment=f"LLM 응답 파싱 오류: {response_str}",
                patent_info="파싱 오류", tech_stack_summary="파싱 오류",
                evaluation_confidence_score=0.1, # 파싱 오류 시 낮은 신뢰도
                sources=all_sources_for_this_evaluation
            )

    except Exception as e:
        print(f"LLM 호출 중 오류 발생: {e}")
        tech_eval_output = TechEvaluationOutput(
            originality_score=None, originality_comment=f"LLM 호출 오류: {e}",
            feasibility_score=None, feasibility_comment=f"LLM 호출 오류: {e}",
            patent_info="LLM 호출 오류", tech_stack_summary="LLM 호출 오류",
            evaluation_confidence_score=0.1, # LLM 호출 오류 시 낮은 신뢰도
            sources=all_sources_for_this_evaluation
        )

    print(f"--- TechEvaluationAgent 실행 완료: {startup_name} ---")
    print(f"분석 결과: {tech_eval_output}")
    return tech_eval_output

# (main 함수는 테스트용이므로 그대로 두거나, 필요시 신뢰도 점수 출력 확인)

if __name__ == '__main__':
    # TechEvaluationAgent 직접 실행 테스트
    # InfoScraperAgent의 출력 예시 (실제로는 state에서 가져와야 함)
    sample_scraped_data_mago = [
        SourceInfo(source_url_or_doc="https://www.holamago.com/", retrieved_content_snippet="MAGO는 음성 AI 빌더 '오디온'과 음성 AI 기반 정신건강 관리 플랫폼 '카세트'를 개발한 기업입니다. 주요 기술은 음성 인식, 자연어 처리, 감정 분석입니다."),
        SourceInfo(source_url_or_doc="MAGO_news_article_1.com", retrieved_content_snippet="마고, 최근 시리즈 A 투자 유치 성공. AI 음성 분석 기술 고도화 계획 발표.")
    ]
    sample_scraped_data_plang = [
        SourceInfo(source_url_or_doc="https://theplang.com/", retrieved_content_snippet="The Plan G는 AI 기반 초등 영어 회화 앱 '오딩가 잉글리시'를 개발했습니다. 아이들이 AI 캐릭터를 가르치며 영어를 배웁니다."),
        SourceInfo(source_url_or_doc="ThePlanG_review.com", retrieved_content_snippet="오딩가 잉글리시는 구글 API를 사용하여 음성-텍스트 변환 및 AI 캐릭터 응답 기능을 제공합니다.")
    ]

    print("\n--- MAGO 기술 평가 테스트 ---")
    mago_tech_eval = run_tech_evaluator(startup_name="MAGO", scraped_info=sample_scraped_data_mago, use_rag=True)
    # print(json.dumps(mago_tech_eval, indent=2, ensure_ascii=False)) # TypedDict를 JSON으로 예쁘게 출력

    print("\n--- The Plan G 기술 평가 테스트 ---")
    plang_tech_eval = run_tech_evaluator(startup_name="The Plan G", scraped_info=sample_scraped_data_plang, use_rag=False) # The Plan G는 RAG 사용 안함 테스트
    # print(json.dumps(plang_tech_eval, indent=2, ensure_ascii=False))