import os

from openai import OpenAI

from python_service.db import fetch_pill_context


class ChatbotService:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for chatbot integration.")
        self.model = os.getenv("SAFEPILL_CHAT_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key)

    def answer(self, question: str, identified_pills: list[str]) -> dict[str, object]:
        if not question.strip():
            raise ValueError("question is required.")

        pill_context = fetch_pill_context(identified_pills)
        context_text = self._format_context(pill_context)
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 의약품 안전 도우미다. 제공된 약품 정보에 근거해서만 답변하고, "
                        "확실하지 않은 정보는 모른다고 말해라. 응급상황 가능성이 있으면 즉시 의료기관 방문을 권고해라."
                    ),
                },
                {
                    "role": "user",
                    "content": f"[식별된 약 정보]\n{context_text}\n\n[질문]\n{question}",
                },
            ],
        )
        answer = completion.choices[0].message.content
        if not answer:
            raise RuntimeError("Chat completion returned empty content.")
        return {"answer": answer, "referencedPills": [item["pillName"] for item in pill_context]}

    def _format_context(self, pill_context: list[dict[str, object]]) -> str:
        if not pill_context:
            return "식별된 약 정보가 없습니다."

        lines: list[str] = []
        for pill in pill_context:
            lines.append(f"- 약품명: {pill['pillName']}")
            if pill.get("manufacturer"):
                lines.append(f"  제조사: {pill['manufacturer']}")
            if pill.get("dosageForm"):
                lines.append(f"  제형: {pill['dosageForm']}")
            ingredients = pill.get("ingredients", [])
            if ingredients:
                ing_text = ", ".join(
                    f"{item['name']}({item['strengthText']})" if item.get("strengthText") else str(item["name"])
                    for item in ingredients
                )
                lines.append(f"  성분: {ing_text}")
            warnings = pill.get("warnings", [])
            if warnings:
                lines.append("  주의사항:")
                for warning in warnings:
                    lines.append(f"    - {warning['text']}")
            interactions = pill.get("interactions", [])
            if interactions:
                lines.append("  상호작용:")
                for interaction in interactions:
                    lines.append(
                        f"    - 대상: {interaction['targetIngredient']}, 위험도: {interaction['severity']}, 내용: {interaction['text']}"
                    )
        return "\n".join(lines)
