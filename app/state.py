# app/state.py

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal, Optional
from langgraph.graph import MessagesState


class PatentResult(TypedDict, total=False):
    app_num: str                  # 출원번호
    title: str                    # 발명의 명칭
    abstract: str                 # 초록 (요약본이면 더 좋음)
    ipc_codes: List[str]          # 주요 IPC 코드들
    score: float                  # 유사도 점수 (0~1)
    # 필요하면 여기에 더 필드 추가 가능 (예: applicant, year 등)


class PatentSearchRecord(TypedDict, total=False):
    id: str                       # "ps_1", "ps_2" 같은 식별자
    query: str                    # 검색에 사용한 정제된 질의 (키워드/문장)
    top_k: int                    # 요청한 결과 개수
    results: List[PatentResult]   # 검색 결과 리스트


class IPCResult(TypedDict, total=False):
    ipc_code: str                 # "G06T 7/00" 등
    title: str                    # IPC 제목
    description: str              # IPC 설명 (짧은 요약)
    score: float | None           # 추천일 때만 의미 있는 점수, 설명일 땐 None 가능


class IPCRequestRecord(TypedDict, total=False):
    id: str                       # "ipc_1", "ipc_2" ...
    mode: Literal["tech_to_ipc", "code_explain"]
    input_text: Optional[str]     # 기술 설명 (mode가 tech_to_ipc일 때)
    codes: List[str]              # 설명을 요청한 IPC 코드들
    top_k: int                    # 추천 개수 또는 검색 개수
    results: List[IPCResult]      # IPC 결과 리스트


class MetaState(TypedDict, total=False):
    turn_index: int               # 몇 번째 턴인지
    last_used_tools: List[str]    # 이번 턴에 어떤 툴 썼는지 (["search_patent", ...])
    session_id: Optional[str]     # 나중에 세션 추적용


class OtherNoteRecord(TypedDict, total=False):
    id: str
    summary: str      # 사용자의 기타 질문 요약
    detail: str       # 조금 더 긴 설명이나 메모


class State(MessagesState):
    """
    LangGraph에서 사용할 전역 상태 정의.
    MessagesState를 상속하면 자동으로 `messages` 필드가 들어있음.
    """
    short_summary: Optional[str]
    patent_history: List[PatentSearchRecord]
    ipc_history: List[IPCRequestRecord]
    other_notes: List[OtherNoteRecord]
    meta: MetaState

# messages:
# LangGraph 기본 → user/assistant 전체 대화 저장

# short_summary:
# 오랫동안 대화하면 MemoryNode가 요약해서 여기 넣음

# patent_history:
# 여러 번의 특허 검색 기록
# 마지막은 patent_history[-1]로 접근

# ipc_history:
# 여러 번의 IPC 요청/설명 기록

# other_notes:
# “이번 턴에서 답해야 하는 기타 사항 요약” 정도로 사용 가능

# meta:
# 턴 인덱스, 최근 사용 툴, 세션 ID 등 운영/디버깅