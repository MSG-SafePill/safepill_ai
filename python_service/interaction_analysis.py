import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


class InteractionAnalysisService:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.offline_mode = os.getenv("SAFEPILL_INTERACTION_OFFLINE", "false").lower() == "true"
        self.model = os.getenv("SAFEPILL_INTERACTION_MODEL", os.getenv("SAFEPILL_CHAT_MODEL", "gpt-4o-mini"))
        self.client = OpenAI(api_key=api_key) if api_key and not self.offline_mode else None
        self.knowledge_path = Path(
            os.getenv(
                "SAFEPILL_MEDICAL_KNOWLEDGE_PATH",
                str(Path(__file__).resolve().parent.parent / "medical_knowledge.txt"),
            )
        )

    def analyze(
        self,
        items: list[dict[str, Any]],
        interaction_rules: list[dict[str, Any]],
        user_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(items) < 2:
            return {
                "riskLevel": "NONE",
                "summary": "분석할 약품 또는 영양제가 2개 미만입니다.",
                "warnings": [],
                "recommendations": ["새로운 약이나 영양제를 추가하기 전에는 의사 또는 약사와 상담하세요."],
                "evidence": [],
                "disclaimer": self._disclaimer(),
            }

        if self.client is None:
            return self._rule_based_response(items, interaction_rules)

        context = self._build_context(items, interaction_rules, user_profile or {})
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 SafePill의 의약품 안전 분석 엔진이다. 제공된 약품/영양제 정보와 "
                        "상호작용 룰, 참고 지식에 근거해서만 답변한다. 진단이나 처방을 하지 말고, "
                        "위험 가능성이 있으면 의사 또는 약사 상담을 권고한다. 반드시 JSON 객체만 반환한다."
                    ),
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
        )

        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Interaction analysis returned empty content.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Interaction analysis returned invalid JSON.") from exc

        return self._normalize_response(parsed)

    def _build_context(
        self,
        items: list[dict[str, Any]],
        interaction_rules: list[dict[str, Any]],
        user_profile: dict[str, Any],
    ) -> str:
        return "\n\n".join(
            [
                "[출력 JSON 스키마]",
                json.dumps(
                    {
                        "riskLevel": "NONE|CAUTION|WARNING|DANGER",
                        "summary": "일반인이 이해하기 쉬운 전체 요약",
                        "warnings": [
                            {
                                "title": "경고 제목",
                                "severity": "CAUTION|WARNING|DANGER",
                                "items": ["관련 약품명 또는 영양제명"],
                                "reason": "왜 위험할 수 있는지",
                            }
                        ],
                        "recommendations": ["사용자가 취할 수 있는 안전한 다음 행동"],
                        "evidence": [
                            {
                                "source": "DUR_RULE|ITEM_PRECAUTION|MEDICAL_KNOWLEDGE",
                                "text": "근거 요약",
                            }
                        ],
                        "disclaimer": self._disclaimer(),
                    },
                    ensure_ascii=False,
                ),
                "[사용자 프로필]",
                json.dumps(user_profile, ensure_ascii=False),
                "[복용/보유 항목]",
                json.dumps(items, ensure_ascii=False),
                "[백엔드 상호작용 룰]",
                json.dumps(interaction_rules, ensure_ascii=False),
                "[참고 지식]",
                self._load_relevant_knowledge(items, interaction_rules),
                "[요청]",
                (
                    "위 정보를 바탕으로 약물 상호작용 위험을 한국어로 분석해라. "
                    "상호작용 룰이 없으면 단정하지 말고 확인된 위험이 없다고 표현해라. "
                    "응급 위험 가능성, 중복 성분, 주의사항이 있으면 명확히 표시해라."
                ),
            ]
        )

    def _load_relevant_knowledge(
        self,
        items: list[dict[str, Any]],
        interaction_rules: list[dict[str, Any]],
    ) -> str:
        if not self.knowledge_path.exists():
            return "참고 지식 파일이 없습니다."

        text = self.knowledge_path.read_text(encoding="utf-8", errors="ignore")
        if "여기에 실제 의학 지식" in text or not text.strip():
            text = self._default_knowledge()
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        keywords = self._keywords(items, interaction_rules)
        scored_blocks: list[tuple[int, str]] = []
        for block in blocks:
            lower_block = block.lower()
            score = sum(1 for keyword in keywords if keyword in lower_block)
            if score > 0:
                scored_blocks.append((score, block))

        selected = [block for _, block in sorted(scored_blocks, key=lambda item: item[0], reverse=True)[:8]]
        if not selected:
            selected = blocks[:3]
        return "\n\n".join(selected)[:6000] if selected else "참고 지식이 비어 있습니다."

    def _keywords(self, items: list[dict[str, Any]], interaction_rules: list[dict[str, Any]]) -> list[str]:
        keywords: set[str] = set()
        for item in items:
            self._add_keyword(keywords, item.get("itemName"))
            for ingredient in item.get("ingredients", []) or []:
                if isinstance(ingredient, dict):
                    self._add_keyword(keywords, ingredient.get("name"))
                else:
                    self._add_keyword(keywords, ingredient)
        for rule in interaction_rules:
            self._add_keyword(keywords, rule.get("ingredientNameA"))
            self._add_keyword(keywords, rule.get("ingredientNameB"))
            self._add_keyword(keywords, rule.get("description"))
        return [keyword.lower() for keyword in keywords if len(keyword) >= 2]

    def _add_keyword(self, keywords: set[str], value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text:
            keywords.add(text)

    def _normalize_response(self, parsed: dict[str, Any]) -> dict[str, Any]:
        risk_level = str(parsed.get("riskLevel") or "NONE").upper()
        if risk_level not in {"NONE", "CAUTION", "WARNING", "DANGER"}:
            risk_level = "CAUTION"

        return {
            "riskLevel": risk_level,
            "summary": str(parsed.get("summary") or "분석 결과 요약을 생성하지 못했습니다."),
            "warnings": parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else [],
            "recommendations": parsed.get("recommendations") if isinstance(parsed.get("recommendations"), list) else [],
            "evidence": parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else [],
            "disclaimer": str(parsed.get("disclaimer") or self._disclaimer()),
        }

    def _rule_based_response(
        self,
        items: list[dict[str, Any]],
        interaction_rules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not interaction_rules:
            return {
                "riskLevel": "NONE",
                "summary": "현재 제공된 상호작용 룰 기준으로 확인된 병용금기 또는 주의 상호작용은 없습니다.",
                "warnings": [],
                "recommendations": [
                    "새로운 약이나 영양제를 추가하기 전에는 의사 또는 약사에게 현재 복용 목록을 보여주세요.",
                    "증상 변화, 알레르기 반응, 출혈, 심한 어지러움 등이 있으면 즉시 전문가에게 문의하세요.",
                ],
                "evidence": [],
                "disclaimer": self._disclaimer(),
            }

        warnings: list[dict[str, Any]] = []
        evidence: list[dict[str, str]] = []
        highest = "CAUTION"
        for rule in interaction_rules:
            severity = str(rule.get("riskLevel") or "CAUTION").upper()
            if severity not in {"CAUTION", "WARNING", "DANGER"}:
                severity = "CAUTION"
            highest = self._max_risk(highest, severity)
            item_names = [
                str(name)
                for name in [rule.get("itemNameA"), rule.get("itemNameB")]
                if name
            ]
            title = f"{rule.get('ingredientNameA')} + {rule.get('ingredientNameB')} 병용 주의"
            description = str(rule.get("description") or "상호작용 가능성이 있어 주의가 필요합니다.")
            warnings.append(
                {
                    "title": title,
                    "severity": severity,
                    "items": item_names,
                    "reason": description,
                }
            )
            evidence.append({"source": "DUR_RULE", "text": description})

        item_count = len(items)
        return {
            "riskLevel": highest,
            "summary": f"총 {item_count}개 항목에서 {len(warnings)}건의 상호작용 주의 항목이 확인되었습니다.",
            "warnings": warnings,
            "recommendations": [
                "복용을 임의로 중단하거나 용량을 바꾸지 말고 의사 또는 약사와 상담하세요.",
                "같은 시간대에 함께 복용 중이라면 상담 전까지 복용 시간 조정이 필요한지 확인하세요.",
                "출혈, 호흡곤란, 심한 발진, 의식 저하 같은 증상이 있으면 즉시 의료기관을 방문하세요.",
            ],
            "evidence": evidence,
            "disclaimer": self._disclaimer(),
        }

    def _max_risk(self, current: str, candidate: str) -> str:
        order = {"NONE": 0, "CAUTION": 1, "WARNING": 2, "DANGER": 3}
        return candidate if order[candidate] > order[current] else current

    def _default_knowledge(self) -> str:
        return (
            "항응고제, 항혈소판제, 오메가3, 은행잎 추출물 등은 출혈 경향과 관련된 주의가 필요할 수 있다. "
            "멍, 코피, 혈뇨, 흑색변 같은 증상이 있으면 의료진 상담이 필요하다.\n\n"
            "진통소염제(NSAIDs)는 위장관 출혈, 신장 부담, 혈압 상승과 관련될 수 있다. "
            "위궤양 병력, 신장질환, 고령자는 전문가 상담이 중요하다.\n\n"
            "중추신경계에 작용하는 약물이나 수면 보조 성분을 함께 복용하면 졸림, 어지러움, 낙상 위험이 커질 수 있다.\n\n"
            "간에서 대사되는 약물은 음주, 간질환, 일부 건강기능식품과 함께 복용할 때 주의가 필요하다.\n\n"
            "철분, 칼슘, 마그네슘 같은 미네랄은 일부 항생제나 갑상선 약의 흡수를 방해할 수 있어 복용 시간 간격이 필요할 수 있다."
        )

    def _disclaimer(self) -> str:
        return "이 분석은 참고용이며 진단이나 처방이 아닙니다. 복용 변경 전 의사 또는 약사와 상담하세요."
