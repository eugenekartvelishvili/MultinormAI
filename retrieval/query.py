from typing import Optional, Union, List
from dataclasses import dataclass

@dataclass
class SearchQuery:
    text: str
    mode: str = "hybrid"              # "dense", "sparse", "hybrid"
    level: Optional[Union[int, List[int]]] = None
    limit: int = 10
    use_summary: bool = False
    dense_weight: float = 0.6         # вес основного dense вектора
    summary_weight: float = 0.2       # вес summary dense вектора
    sparse_weight: float = 0.2        # вес sparse вектора
    filter_expr: Optional[str] = None # дополнительный фильтр, например doc_id == '...'

    @property
    def levels(self) -> Optional[List[int]]:
        if self.level is None:
            return None
        if isinstance(self.level, int):
            return [self.level]
        return self.level
