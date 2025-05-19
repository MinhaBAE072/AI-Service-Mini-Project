# AI-Service-Mini-Project-main/agents/report_generator_agent.py
import sys
import os
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리를 sys.path에 추가
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

# .env 파일 로드
env_path = os.path.join(PROJECT_ROOT_DIR, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    print(f"경고: .env 파일을 찾을 수 없습니다. ({env_path}) OpenAI API 키가 환경 변수에 설정되어 있어야 합니다.")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from states.ai_mini_design_20_3반_배민하_state_py import (
    AdvancedStartupEvaluationState,
    SingleStartupDetailedAnalysis,
    # 테스트용 샘플 데이터 생성을 위해 다른 Output TypedDict도 임포트
    TechEvaluationOutput, MarketEvaluationOutput, TeamEvaluationOutput, BizModelEvaluationOutput, SourceInfo
)

# LLM 초기화 (보고서 생성은 긴 출력이 필요할 수 있음)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, model_kwargs={"max_tokens": 4095}) # GPT-4o-mini의 최대 토큰 고려

# 보고서 생성 프롬프트 파일 경로 (사용자가 제공한 파일명으로 수정)
PROMPT_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "prompts", "ai_mini_design_20_3반_배민하_prompt.md") # 파일명 수정

def load_prompt_template_from_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"오류: 프롬프트 파일을 찾을 수 없습니다 - {file_path}")
        return "보고서 생성 프롬프트 로드 실패."

def run_report_generator(state: AdvancedStartupEvaluationState) -> Dict[str, str]:
    print(f"--- ReportGeneratorAgent 실행 시작 ---")
    updates_to_state: Dict[str, str] = {}
    current_error_log = state.get("error_log", [])

    individual_analyses: List[SingleStartupDetailedAnalysis] = state.get("individual_detailed_analyses", [])
    comparative_output: Dict[str, Any] = state.get("comparative_analysis_output", {})
    startup_names_to_compare: List[str] = state.get("startup_names_to_compare", [])

    if not individual_analyses or not comparative_output or not startup_names_to_compare:
        error_msg = "보고서 생성을 위한 개별 분석, 비교 분석, 또는 스타트업 이름 정보가 부족합니다."
        print(error_msg)
        updates_to_state["final_comparison_report_text"] = f"보고서 생성 실패: {error_msg}"
        updates_to_state["error_log"] = current_error_log + [f"ReportGeneratorAgent: {error_msg}"]
        return updates_to_state

    try:
        # 프롬프트에 삽입하기 위해 데이터를 JSON 문자열로 변환
        # ensure_ascii=False로 한글 유지, TypedDict는 dict()로 변환 후 직렬화
        individual_analyses_json_str = json.dumps([dict(analysis) for analysis in individual_analyses], indent=2, ensure_ascii=False)
        comparative_output_json_str = json.dumps(comparative_output, indent=2, ensure_ascii=False)
    except TypeError as e:
        error_msg = f"데이터 JSON 직렬화 오류: {e}"
        print(error_msg)
        updates_to_state["final_comparison_report_text"] = f"보고서 생성 실패: {error_msg}"
        updates_to_state["error_log"] = current_error_log + [f"ReportGeneratorAgent: {error_msg}"]
        return updates_to_state

    report_base_prompt_str = load_prompt_template_from_file(PROMPT_FILE_PATH)
    if "로드 실패" in report_base_prompt_str:
        updates_to_state["final_comparison_report_text"] = report_base_prompt_str
        updates_to_state["error_log"] = current_error_log + ["ReportGeneratorAgent: 프롬프트 파일 로드 실패"]
        return updates_to_state

    # 프롬프트 파일 내의 플레이스홀더에 맞게 입력 변수 준비
    # 예: {startup_names_to_compare[0]}, {len(startup_names_to_compare)}, {startup_names_to_compare[2:]}
    # {individual_detailed_analyses}, {comparative_analysis_output}
    
    # LangChain의 PromptTemplate을 사용할 경우, input_variables를 명시적으로 정의해야 합니다.
    # 프롬프트 파일 내의 리스트 인덱싱, len() 호출 등은 f-string 이나 .format()으로 사전 처리 필요.

    # 입력 변수 준비
    template_vars = {
        "individual_detailed_analyses": individual_analyses_json_str,
        "comparative_analysis_output": comparative_output_json_str,
        "startup_names_to_compare_0": startup_names_to_compare[0] if len(startup_names_to_compare) > 0 else "N/A",
        "startup_names_to_compare_1": startup_names_to_compare[1] if len(startup_names_to_compare) > 1 else "N/A",
        "len_startup_names_to_compare": str(len(startup_names_to_compare)), # 문자열로 변환
        "startup_names_to_compare_2_onwards": ", ".join(startup_names_to_compare[2:]) if len(startup_names_to_compare) > 2 else "없음"
    }

    # 프롬프트 파일 내의 플레이스홀더를 위 template_vars의 키와 일치하도록 수정해야 합니다.
    # 예: [{startup_names_to_compare[0]}] -> [{startup_0}] 또는 여기서 직접 문자열 포매팅
    # 여기서는 프롬프트 파일 내용을 읽어와서 .replace()로 단순 치환 시도 (더 나은 방법은 PromptTemplate 사용)

    # 프롬프트 파일의 내용을 기반으로 final_prompt_for_llm 구성
    # ai_mini_design_20_3반_배민하_prompt.md 의 변수 형식에 맞춤
    final_prompt_for_llm = report_base_prompt_str.replace(
        "[{startup_names_to_compare[0]}]", f"[{template_vars['startup_names_to_compare_0']}]"
    ).replace(
        "[{startup_names_to_compare[1]}]", f"[{template_vars['startup_names_to_compare_1']}]"
    ).replace(
        "{len(startup_names_to_compare)}", template_vars['len_startup_names_to_compare']
    ).replace(
        "{startup_names_to_compare[2:]}", f"({template_vars['startup_names_to_compare_2_onwards']})" # 괄호 추가하여 명시
    ).replace(
        "{individual_detailed_analyses}", template_vars['individual_detailed_analyses'] # 이미 JSON 문자열임
    ).replace(
        "{comparative_analysis_output}", template_vars['comparative_analysis_output'] # 이미 JSON 문자열임
    )
    
    # 보고서 제목의 플레이스홀더도 처리
    final_prompt_for_llm = final_prompt_for_llm.replace(
        "AI 스타트업 비교 분석 보고서: {startup_names_to_compare[0]} vs {startup_names_to_compare[1]} (vs ...)",
        f"AI 스타트업 비교 분석 보고서: {template_vars['startup_names_to_compare_0']} vs {template_vars['startup_names_to_compare_1']}" +
        (f" vs {template_vars['startup_names_to_compare_2_onwards']}" if template_vars['startup_names_to_compare_2_onwards'] != "없음" else "")
    )
    # 제1부, 제2부 등의 제목 플레이스홀더 처리
    final_prompt_for_llm = final_prompt_for_llm.replace(
        "제1부: [{startup_names_to_compare[0]}] 심층 분석",
        f"제1부: [{template_vars['startup_names_to_compare_0']}] 심층 분석"
    ).replace(
        "제2부: [{startup_names_to_compare[1]}] 심층 분석",
        f"제2부: [{template_vars['startup_names_to_compare_1']}] 심층 분석"
    )
    # 필요시 제N부도 유사하게 처리 (루프 사용 또는 프롬프트 수정)

    # print(f"LLM에 전달될 최종 보고서 생성 프롬프트 (일부):\n{final_prompt_for_llm[:1000]}...") # 로그 필요시

    try:
        response = llm.invoke(final_prompt_for_llm)
        final_report_text = response.content
        updates_to_state["final_comparison_report_text"] = final_report_text
        print(f"--- ReportGeneratorAgent 실행 완료 ---")
        print(f"생성된 보고서 (일부): {final_report_text[:500]}...")
    except Exception as e:
        error_msg = f"LLM 호출 중 보고서 생성 오류 발생: {e}"
        print(error_msg)
        updates_to_state["final_comparison_report_text"] = f"보고서 생성 실패: {error_msg}"
        updates_to_state["error_log"] = current_error_log + [f"ReportGeneratorAgent: {error_msg}"]
        
    return updates_to_state

if __name__ == '__main__':
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    else:
        # 테스트용 샘플 데이터 (이전 에이전트 테스트에서 가져옴)
        sample_mago_analysis = SingleStartupDetailedAnalysis(
            startup_name="MAGO", company_overview="MAGO는 음성 AI 및 정신건강 관리 플랫폼 개발 스타트업입니다...",
            tech_evaluation=TechEvaluationOutput(originality_score=4.0, originality_comment="음성 AI와 정신건강 결합 독창적", feasibility_score=4.5, feasibility_comment="기술 구현 가능성 높음", patent_info="정보 부족", tech_stack_summary="음성인식, NLP", sources=[]),
            market_evaluation=MarketEvaluationOutput(market_size_score=4.2, market_size_comment="디지털 정신건강 시장 매력적", growth_potential_score=4.5, growth_potential_comment="AI 기반 솔루션 수요 증가", competitive_landscape_summary="경쟁 강도 높으나 차별화 가능", entry_barriers_comment="기술 장벽 높음", network_effects_comment="네트워크 효과 존재", sources=[]),
            team_evaluation=TeamEvaluationOutput(founder_expertise_score=4.3, founder_expertise_comment="CEO 음성 AI 분야 경험 풍부", founder_communication_score=4.0, founder_communication_comment="비전 전달 양호", founder_execution_score=4.1, founder_execution_comment="초기 성과 있음", key_team_members_summary="Sukbong Kwon", sources=[]),
            biz_model_evaluation=BizModelEvaluationOutput(track_record_summary="시드 투자 유치 준비 중", track_record_score=3.0, deal_terms_comment="정보 부족", deal_terms_score=None, revenue_model_summary="B2B, B2B2C, 구독 모델", g2m_strategy_summary="Naver 협력", sources=[]),
            weighted_overall_score=4.1, strength_summary="기술력, 성장 잠재력", weakness_summary="초기 실적 부족", opportunity_summary="정신건강 시장 확대", threat_summary="대기업 경쟁 심화", scraped_urls=[], key_documents_retrieved=[]
        )
        sample_plang_analysis = SingleStartupDetailedAnalysis(
            startup_name="The Plan G", company_overview="The Plan G는 AI 기반 초등 영어 학습 앱 '오딩가 잉글리시' 개발사...",
            tech_evaluation=TechEvaluationOutput(originality_score=3.8, originality_comment="Teach-to-learn 방식 신선", feasibility_score=4.0, feasibility_comment="구현 용이", patent_info="정보 부족", tech_stack_summary="GenAI, Google API", sources=[]),
            market_evaluation=MarketEvaluationOutput(market_size_score=4.0, market_size_comment="에듀테크 초등 영어 시장 큼", growth_potential_score=4.3, growth_potential_comment="비대면 학습 수요 증가", competitive_landscape_summary="유사 앱 다수, 경쟁 치열", entry_barriers_comment="콘텐츠, 마케팅 비용", network_effects_comment="사용자 데이터 축적시 모델 개선", sources=[]),
            team_evaluation=TeamEvaluationOutput(founder_expertise_score=4.2, founder_expertise_comment="CEO 교육 콘텐츠 경력", founder_communication_score=3.9, founder_communication_comment="정보 부족 추론", founder_execution_score=4.2, founder_execution_comment="인도 시장 진출 성과", key_team_members_summary="정보 부족", sources=[]),
            biz_model_evaluation=BizModelEvaluationOutput(track_record_summary="누적 다운로드 15만, 시드 2억", track_record_score=3.8, deal_terms_comment="시드 투자 완료", deal_terms_score=3.5, revenue_model_summary="Freemium, B2B 라이선스", g2m_strategy_summary="온라인 마케팅, 학교 파트너십", sources=[]),
            weighted_overall_score=3.9, strength_summary="시장 실적, 실행력", weakness_summary="기술 독창성 상대적 부족", opportunity_summary="글로벌 확장", threat_summary="경쟁 심화", scraped_urls=[], key_documents_retrieved=[]
        )
        sample_comparative_output = {
            "comparison_summary_text": "MAGO는 기술 혁신성과 성장 잠재력에서, The Plan G는 현재 시장 실적과 명확한 수익 모델에서 강점을 보입니다. 투자 성향에 따라 선택이 달라질 수 있습니다.",
            "side_by_side_comparison_data": {
                "기술_독창성_점수": {"MAGO": 4.0, "The Plan G": 3.8},
                "시장_성장잠재력_점수": {"MAGO": 4.5, "The Plan G": 4.3},
                "팀_실행력_점수_추론": {"MAGO": "초기 단계, PoC 진행 중", "The Plan G": "인도 시장 진출 등 구체적 성과"},
                "사업모델_주요실적_요약": {"MAGO": "시드 투자 유치 준비 중", "The Plan G": "누적 다운로드 15만"}
            },
            "overall_recommendation": "기술 혁신을 중시하는 초기 투자자는 MAGO, 안정적인 시장과 명확한 실적을 선호하는 투자자는 The Plan G를 고려할 수 있습니다."
        }

        test_state_for_report = AdvancedStartupEvaluationState(
            startup_names_to_compare=["MAGO", "The Plan G"],
            individual_detailed_analyses=[sample_mago_analysis, sample_plang_analysis],
            comparative_analysis_output=sample_comparative_output,
            messages=[],
            error_log=[]
        )

        report_output_update = run_report_generator(test_state_for_report)
        
        print("\n--- 최종 보고서 생성 결과 (final_comparison_report_text 일부) ---")
        final_report = report_output_update.get("final_comparison_report_text", "보고서 생성 실패 또는 오류")
        if "실패" not in final_report :
            print(final_report[:2000] + "...") # 보고서가 길 수 있으므로 일부만 출력
            # 파일로 저장하고 싶다면:
            # with open("generated_report.md", "w", encoding="utf-8") as f:
            #     f.write(final_report)
            # print("\n보고서가 generated_report.md 파일로 저장되었습니다.")
        else:
            print(final_report)

        if report_output_update.get("error_log"):
            print("\n--- 오류 로그 ---")
            for error in report_output_update["error_log"]:
                print(error)