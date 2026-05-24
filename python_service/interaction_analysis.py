import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


class InteractionAnalysisService:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for interaction analysis.")
        self.model = os.getenv("SAFEPILL_INTERACTION_MODEL", os.getenv("SAFEPILL_CHAT_MODEL", "gpt-4o-mini"))
        self.client = OpenAI(api_key=api_key)
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
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        keywords = self._keywords(items, interaction_rules)
        selected: list[str] = []
        for block in blocks:
            lower_block = block.lower()
            if any(keyword in lower_block for keyword in keywords):
                selected.append(block)
            if len(selected) >= 8:
                break

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

    def _disclaimer(self) -> str:
        return "이 분석은 참고용이며 진단이나 처방이 아닙니다. 복용 변경 전 의사 또는 약사와 상담하세요."
