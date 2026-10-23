"""Per-conversation history and context-window management.

Strategy
--------
Every conversation keeps its full message list plus a `summary` string.
Before each request we build the context that is actually sent to the model:

* truncate  - send only the last KEEP_RECENT messages (older ones are dropped).
* summarize - when history grows past MAX_MESSAGES, the oldest messages are
              folded into a running summary by an extra LLM call; the summary
              is prepended to the system prompt and only KEEP_RECENT messages
              are sent verbatim. If summarising fails we fall back to truncation,
              so a summary error never breaks the chat.
"""
import threading
import uuid
from dataclasses import dataclass, field

from .config import settings
from .llm import LLMClient, LLMError
from .prompts import SUMMARY_PROMPT, SYSTEM_PROMPT


@dataclass
class Conversation:
    id: str
    messages: list[dict] = field(default_factory=list)  # messages not yet summarised
    summary: str = ""
    total_turns: int = 0


class ConversationManager:
    def __init__(
        self,
        llm: LLMClient,
        max_messages: int = settings.max_messages,
        keep_recent: int = settings.keep_recent,
        strategy: str = settings.context_strategy,
    ):
        if keep_recent < 2 or keep_recent >= max_messages:
            raise ValueError("Need 2 <= KEEP_RECENT < MAX_MESSAGES")
        if keep_recent % 2:  # keep whole user/assistant pairs
            keep_recent -= 1
        self.llm = llm
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.strategy = strategy
        self._conversations: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    # --- storage -----------------------------------------------------------
    def get_or_create(self, conversation_id: str | None) -> Conversation:
        with self._lock:
            if conversation_id and conversation_id in self._conversations:
                return self._conversations[conversation_id]
            conv = Conversation(id=conversation_id or uuid.uuid4().hex)
            self._conversations[conv.id] = conv
            return conv

    def reset(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)

    # --- context management ------------------------------------------------
    def _compact(self, conv: Conversation) -> None:
        """Shrink history once it exceeds max_messages."""
        if len(conv.messages) <= self.max_messages:
            return
        old = conv.messages[: -self.keep_recent]
        recent = conv.messages[-self.keep_recent :]

        if self.strategy == "summarize":
            transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in old)
            prompt = SUMMARY_PROMPT.format(summary=conv.summary or "(none yet)", messages=transcript)
            try:
                conv.summary = self.llm.generate(
                    "You write accurate, compact conversation summaries.",
                    [{"role": "user", "content": prompt}],
                )
            except LLMError:
                pass  # fall back to plain truncation; old summary is kept
        conv.messages = recent

    def build_system_prompt(self, conv: Conversation) -> str:
        if not conv.summary:
            return SYSTEM_PROMPT
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Summary of the earlier part of this conversation (use it as memory):\n{conv.summary}"
        )

    # --- main entry point --------------------------------------------------
    def chat(self, conversation_id: str | None, user_message: str) -> tuple[Conversation, str]:
        conv = self.get_or_create(conversation_id)
        candidate = conv.messages + [{"role": "user", "content": user_message}]

        # Raises LLMError on failure; history is only updated on success so a
        # failed call can simply be retried.
        reply = self.llm.generate(self.build_system_prompt(conv), candidate)

        conv.messages = candidate + [{"role": "assistant", "content": reply}]
        conv.total_turns += 1
        self._compact(conv)
        return conv, reply
