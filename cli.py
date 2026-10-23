"""Terminal chat interface (no server needed).

Usage:  python cli.py            commands: /reset  /debug  /quit
"""
from app.conversation import ConversationManager
from app.llm import GeminiClient, LLMError


def main() -> None:
    try:
        manager = ConversationManager(llm=GeminiClient())
    except LLMError as e:
        print(f"Error: {e.user_message}")
        return

    conv_id = None
    print("Mentor chatbot - type /reset, /debug or /quit\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            if conv_id:
                manager.reset(conv_id)
            conv_id = None
            print("(conversation cleared)\n")
            continue
        if text == "/debug":
            if conv_id:
                conv = manager.get_or_create(conv_id)
                print(f"(messages in context: {len(conv.messages)}, summary: {conv.summary or 'none'})\n")
            continue
        try:
            conv, reply = manager.chat(conv_id, text)
            conv_id = conv.id
            print(f"Mentor: {reply}\n")
        except LLMError as e:
            print(f"[error] {e.user_message}\n")


if __name__ == "__main__":
    main()
