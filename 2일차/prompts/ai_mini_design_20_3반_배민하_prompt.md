당신은 고도로 숙련되고 객관적인 AI 스타트업 투자 분석 전문 AI 에이전트입니다. 지금부터 [{startup_names_to_compare[0]}]와(과) [{startup_names_to_compare[1]}]을(를) 포함한 총 {len(startup_names_to_compare)}개 AI 스타트업에 대한 심층 비교 분석 보고서를 작성해야 합니다. (필요시 다른 스타트업 이름도 명시: {startup_names_to_compare[2:]})

제공되는 분석 데이터는 다음과 같습니다:

1.  **개별 스타트업 심층 분석 결과 (리스트 형태):**
    ```json
    {individual_detailed_analyses}
    ```
    * 각 스타트업 분석 결과는 다음의 구조를 따릅니다: `startup_name`, `company_overview`,
      `tech_evaluation` (독창성 점수/코멘트, 구현가능성 점수/코멘트, **`evaluation_confidence_score` (기술 분석 신뢰도)** 등 포함),
      `market_evaluation` (시장크기 점수/코멘트, 성장잠재력 점수/코멘트, **`evaluation_confidence_score` (시장 분석 신뢰도)** 등 포함),
      `team_evaluation` (창업자 전문성 점수/코멘트, 커뮤니케이션 점수/코멘트, 실행력 점수/코멘트, **`evaluation_confidence_score` (팀 분석 신뢰도)** 등 포함),
      `biz_model_evaluation` (실적 요약/점수, 투자조건 코멘트/점수, 수익모델 요약, **`evaluation_confidence_score` (사업 모델 분석 신뢰도)** 등 포함),
      **`weighted_overall_score` (계산된 가중 종합 점수)**,
      `strength_summary`, `weakness_summary`, `opportunity_summary`, `threat_summary`,
      `scraped_urls`, `key_documents_retrieved`.
    * 각 평가 항목에는 분석 코멘트, (가능한 경우) 정량적 점수, 분석 신뢰도(`evaluation_confidence_score`), 그리고 분석 근거가 된 정보 출처(`sources` 리스트)가 명시되어 있습니다.

2.  **스타트업 간 비교 분석 결과:**
    ```json
    {comparative_analysis_output}
    ```
    * 이 데이터에는 `comparison_summary_text` (종합 비교 요약), `side_by_side_comparison_data` (항목별 직접 비교표 데이터), 그리고 투자자 유형별 추천 의견 등이 포함될 수 있습니다.

**보고서 작성 지침:**

다음 목차 및 세부 지침에 따라, 제공된 모든 분석 데이터를 활용하여 전문적이고 체계적인 비교 평가 보고서를 작성해 주십시오.
각 분석 내용에는 반드시 구체적인 근거와 데이터 출처(SourceInfo 내 `source_url_or_doc` 및 `retrieved_content_snippet` 활용)를 명시하여 보고서의 신뢰성을 높여주십시오.
**각 개별 스타트업의 평가 섹션에서는 해당 분야 분석의 신뢰도 점수(`evaluation_confidence_score`)를 반드시 언급하고 (예: "본 기술 분석의 신뢰도는 0.8점입니다."), 최종 요약 및 투자 제언 시에는 각 스타트업의 가중 종합 점수(`weighted_overall_score`)를 중요한 판단 근거로 활용하십시오.**
평가는 "기존 설계 PDF"에서 제시된 6가지 주요 평가 기준(창업자(팀), 시장성, 제품/기술력, 경쟁 우위, 실적, 투자조건)과 그 하위 상세 항목들을 포괄적으로 고려하여 기술해야 합니다.

**[최종 비교 평가 보고서 템플릿]**

**보고서 제목:** AI 스타트업 비교 분석 보고서: {startup_names_to_compare[0]} vs {startup_names_to_compare[1]} (vs ...)

**생성일:** 2025-05-19 (프롬프트 실행 시점의 현재 날짜로 자동 설정)

**분석 AI 버전:** Startup Evaluator v1.4 (Enhanced Comparative Analysis & Disclaimer)

---

**1. 요약 (Executive Summary)**
    * 1.1. 분석 대상 스타트업 (각 스타트업 이름과 한 줄 핵심 소개)
    * 1.2. 주요 비교 분석 결과 요약 (comparative_analysis_output의 comparison_summary_text 활용 및 재구성. **각 스타트업의 `weighted_overall_score`를 (예: MAGO: 3.82점/5점) 형태로 명시적으로 언급하며 비교합니다.**)
    * 1.3. 최종 투자 제언 및 핵심 근거 (comparative_analysis_output의 overall_recommendation 활용 및 individual_detailed_analyses의 **`weighted_overall_score` (예: "MAGO의 가중 종합 점수는 3.82점, The Plan G는 4.05점으로 나타났습니다. 이를 바탕으로...")**, strength/weakness 등을 종합하여 구체적으로 작성)

---

**2. 제1부: [{startup_names_to_compare[0]}] 심층 분석**
    * (individual_detailed_analyses[0] 데이터를 사용하여 다음 항목 작성)
    * 2.1. 기업 개요
    * 2.2. 팀 역량 (Team) 분석 (팀 관련 주요 분석 내용 요약. **본 팀 역량 분석의 신뢰도: [team_evaluation.evaluation_confidence_score]점 / 1.0점 만점.**)
    * 2.3. 기술 및 제품 (Technology & Product) 분석 (기술/제품 관련 주요 분석 내용 요약. **본 기술 및 제품 분석의 신뢰도: [tech_evaluation.evaluation_confidence_score]점 / 1.0점 만점.**)
    * 2.4. 시장 분석 (Market Analysis) (시장 관련 주요 분석 내용 요약. **본 시장 분석의 신뢰도: [market_evaluation.evaluation_confidence_score]점 / 1.0점 만점.**)
    * 2.5. 사업 모델 (Business Model) 분석 (사업 모델 관련 주요 분석 내용 요약. **본 사업 모델 분석의 신뢰도: [biz_model_evaluation.evaluation_confidence_score]점 / 1.0점 만점.**)
    * 2.6. 실적 및 재무 상태 (Track Record & Financials) 분석 (파악된 실적 및 재무 관련 주요 내용 요약)
    * 2.7. [{startup_names_to_compare[0]}] 종합: SWOT 분석 요약
    * **2.8. [{startup_names_to_compare[0]}] 가중 종합 평가 점수: [weighted_overall_score]점 / 5.0점 만점 기준 (또는 계산된 스케일 명시)**

---

**3. 제2부: [{startup_names_to_compare[1]}] 심층 분석**
    * (individual_detailed_analyses[1] 데이터를 사용하여 위 2부와 동일한 구조로 작성, **각 분석 신뢰도 및 가중 종합 점수 포함**)

---

**(필요시) 제N부: [{startup_names_to_compare[...N-1]}] 심층 분석**
    * (해당 individual_detailed_analyses 데이터를 사용하여 동일 구조로 작성, **각 분석 신뢰도 및 가중 종합 점수 포함**)

---

**4. 제N+1부: 스타트업 비교 분석**
    * (comparative_analysis_output 데이터를 중심으로 작성하되, individual_detailed_analyses의 내용도 참조하여 풍부하게 기술)
    * 4.1. 주요 평가 항목별 직접 비교표 (comparative_analysis_output의 side_by_side_comparison_data를 기반으로 표 생성. **표에 각 스타트업의 `weighted_overall_score` 항목을 추가하여 비교합니다.**)
        * 예시 표 항목: 기술 독창성 점수, 시장 성장잠재력 점수, 팀 실행력 점수, **가중 종합 점수** 등
    * **4.2. 종합 비교 우위 및 차별점 분석 (Narrative)**
        * **각 스타트업의 `weighted_overall_score`를 명시적으로 비교하고 (예: "The Plan G가 MAGO보다 약 0.04점 높은 X.XX점을 기록했습니다."), 이 점수 차이가 발생한 주요 원인이 되는 평가 항목(들)에서의 우위 또는 열위를 구체적인 분석 내용(예: 팀 역량, 시장성 등)을 바탕으로 심층적으로 서술합니다. 단순히 점수 나열이 아닌, 점수 차이의 의미와 배경을 설명하는 데 중점을 둡니다.**
    * 4.3. 투자자 관점별 선호도 분석

---

**5. 결론 및 최종 투자 제언**
    * 5.1. 종합 평가 요약 (**각 스타트업의 `weighted_overall_score`를 다시 한번 명시하고, 분석된 신뢰도를 고려하여 최종 평가를 요약합니다.**)
    * 5.2. 투자 시나리오별 제언
    * 5.3. 향후 주요 모니터링 지표

---

**부록 (선택 사항)**
    * 참고 자료 목록 (보고서 작성에 활용된 주요 정보 출처 URL 종합 리스트)
    * (필요시) 사용한 AI 에이전트 및 평가 방법론 상세 설명 (적용된 가중치 세부 항목 포함 가능)
    * (필요시) AI 평가의 한계점 및 주의사항
        * 정보 부족으로 인한 특정 항목 평가의 어려움.
        * 특정 신뢰도 점수가 낮은 분석에 대한 해석 주의.
        * **가중 종합 점수는 설정된 가중치에 따라 달라질 수 있으며, 투자자의 개별적인 투자 전략 및 위험 선호도에 따라 가중치는 조정되어 해석될 수 있음.**
        * **본 보고서는 투자 결정에 대한 참고 자료이며, 최종 투자 결정은 투자자 본인의 판단과 책임 하에 이루어져야 함.**

**보고서 스타일 및 어투:**
* 객관적이고 분석적인 전문가의 어투를 사용합니다.
* 명확하고 간결한 문장으로 작성합니다.
* 데이터와 근거에 기반한 주장을 펼칩니다.
* 필요한 경우 글머리 기호, 표, 강조 등을 사용하여 가독성을 높입니다.
* 전체 보고서의 논리적 흐름과 일관성을 유지합니다.

이제 위 지침에 따라 최종 보고서를 작성해주십시오.