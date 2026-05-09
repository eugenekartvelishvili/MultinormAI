from collections import defaultdict
from typing import List, Dict


class ContextBuilder:
    """
    Собирает иерархический контекст:
    H3 → H2 → (опционально) doc
    """

    def build(self, hits: List[dict]) -> str:
        grouped: Dict[str, List[dict]] = defaultdict(list)

        for h in hits:
            key = h.get("main_section_id") or h.get("section_id")
            grouped[key].append(h)

        blocks = []

        for _, group in grouped.items():
            group = sorted(group, key=lambda x: x["level"])

            for h in group:
                prefix = self._prefix(h["level"], h.get("number"))
                blocks.append(f"{prefix} {h['text']}")

        return "\n\n".join(blocks)

    @staticmethod
    def _prefix(level: int, number: str | None):
        if level == 0:
            return f"📄 Документ {number or ''}"
        if level == 1:
            return f"## {number or ''}"
        if level == 2:
            return f"- {number or ''}"
        return ""
