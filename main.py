from dotenv import load_dotenv
load_dotenv()
from utils.audio_processor import process_input
from utils.transcriber import transcribe_audio

from utils.core.summarize import summarize, generate_title
from utils.core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from utils.core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant")

    chunks = process_input(source)

# Pass the original source path, not the list of chunks
    transcript = transcribe_audio(chunks, language=language)

    # Generate title and summary
    title = generate_title(transcript)
    summary = summarize(transcript)

    # Extract meeting information
    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    # Build RAG chain for transcript question-answering
    rag_chain = build_rag_chain(transcript)

    print("AI Video Assistant pipeline completed")

    return {
        "title": title,
        "summary": summary,
        "transcript": transcript,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "questions": questions,
        "rag_chain": rag_chain,
    }


def ask_question(rag_chain, question: str) -> str:
    """Answer a question using the RAG chain."""
    return rag_chain.invoke(question)


if __name__ == "__main__":
    source = input("Enter video/audio file path or URL: ").strip()
    language = input("Enter language (default: english): ").strip() or "english"

    result = run_pipeline(source, language)

    print("\n--- VIDEO TITLE ---")
    print(result["title"])

    print("\n--- SUMMARY ---")
    print(result["summary"])

    print("\n--- ACTION ITEMS ---")
    print(result["action_items"])

    print("\n--- KEY DECISIONS ---")
    print(result["key_decisions"])

    print("\n--- OPEN QUESTIONS ---")
    print(result["questions"])

    print("\nYou can now ask questions about the transcript.")
    while True:
        question = input("\nYour question (or type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            break

        if question:
            answer = ask_question(result["rag_chain"], question)
            print("\nAnswer:", answer)