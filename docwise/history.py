"""Framework-independent conversation state."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


NUMBERED_ITEM_PATTERN = re.compile(r"^\s*(\d{1,2})[.)]\s+(.+?)\s*$", re.MULTILINE)
ORDINAL_PATTERN = re.compile(
    r"\b(?P<article>the\s+)?(?P<ordinal>first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth|\d{1,2}(?:st|nd|rd|th))\s+"
    r"(?P<noun>project|item|example|one)\b",
    re.IGNORECASE,
)
FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|its|that one|this one|those|them|they|their|former|latter|"
    r"previous example|previous project|above|earlier)\b|"
    r"\b(explain|describe|expand on)\b.*\b(further|more detail)\b|"
    r"^(and|also|what about|how about)\b",
    re.IGNORECASE,
)

ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


@dataclass(slots=True)
class ChatTurn:
    role: str
    content: str
    resolved_question: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceResolution:
    question: str
    anchor: str | None = None
    used_history: bool = False


@dataclass(slots=True)
class ConversationHistory:
    turns: list[ChatTurn] = field(default_factory=list)
    max_turns: int = 12

    def add_user(self, content: str) -> None:
        self.turns.append(ChatTurn("user", content))
        self._trim()

    def add_assistant(self, content: str, resolved_question: str | None = None) -> None:
        self.turns.append(ChatTurn("assistant", content, resolved_question))
        self._trim()

    @classmethod
    def from_messages(
        cls,
        messages: Iterable[Mapping[str, Any]],
        *,
        max_turns: int = 12,
    ) -> "ConversationHistory":
        """Rebuild history from UI state while retaining assistant topic metadata."""
        history = cls(max_turns=max_turns)
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"user", "assistant"} or not content:
                continue
            history.turns.append(
                ChatTurn(role, content, message.get("resolved_question"))
            )
        history._trim()
        return history

    def clear(self) -> None:
        self.turns.clear()

    def _trim(self) -> None:
        self.turns[:] = self.turns[-self.max_turns :]

    def formatted(self) -> str:
        parts: list[str] = []
        for turn in self.turns:
            parts.append(f"{turn.role.title()}: {turn.content}")
            if turn.role == "assistant" and turn.resolved_question:
                parts.append(f"Prior standalone topic: {turn.resolved_question}")
        return "\n".join(parts)

    def latest_topic_question(self) -> str | None:
        """Return the last resolved topic, falling back to the last user question."""
        for turn in reversed(self.turns):
            if turn.role == "assistant" and turn.resolved_question:
                return turn.resolved_question
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None

    def most_recent_numbered_items(self) -> dict[int, str]:
        """Extract navigation labels from the latest assistant-created numbered list."""
        for turn in reversed(self.turns):
            if turn.role != "assistant":
                continue
            items: dict[int, str] = {}
            for number, raw_item in NUMBERED_ITEM_PATTERN.findall(turn.content):
                item = _clean_list_label(raw_item)
                if item:
                    items[int(number)] = item
            if items:
                return items
        return {}

    def resolve_reference(self, question: str) -> ReferenceResolution:
        """Resolve explicit ordinals deterministically and provide safe follow-up context."""
        ordinal_match = ORDINAL_PATTERN.search(question)
        if ordinal_match:
            ordinal_text = ordinal_match.group("ordinal").lower()
            position = ORDINALS.get(ordinal_text)
            if position is None:
                position = int(re.match(r"\d+", ordinal_text).group())
            label = self.most_recent_numbered_items().get(position)
            if label:
                noun = ordinal_match.group("noun").lower()
                replacement = label
                if noun == "project" and not re.search(r"\bproject\b", label, re.IGNORECASE):
                    replacement += " project"
                if ordinal_match.group("article"):
                    replacement = f"the {replacement}"
                resolved = question[: ordinal_match.start()] + replacement + question[ordinal_match.end() :]
                return ReferenceResolution(resolved, anchor=label, used_history=True)

        if FOLLOW_UP_PATTERN.search(question):
            topic = self.latest_topic_question()
            if topic:
                return ReferenceResolution(
                    f"{question} Refer to the subject of this prior standalone question: {topic}",
                    anchor=None,
                    used_history=True,
                )
        return ReferenceResolution(question)


def _clean_list_label(raw_item: str) -> str:
    text = raw_item.strip()
    bold = re.match(r"\*\*(.+?)\*\*", text)
    if bold:
        text = bold.group(1)
    else:
        text = re.split(r"\s+(?:—|–|-)\s+|:\s+", text, maxsplit=1)[0]
    text = re.sub(r"\s*\[[0-9, ]+\]\s*$", "", text)
    return text.strip(" *_`#.-")
