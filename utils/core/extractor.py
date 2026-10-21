import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    """Initialize the Groq language model."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model="qwen/qwen3.8-27b",
        api_key=api_key,
        temperature=0.3,
    )


def build_chain(llm, system_prompt: str):
    """Build a reusable LangChain prompt pipeline."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x["text"]})
        | prompt
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """Extract action items from a meeting transcript."""

    if not transcript or not transcript.strip():
        return "No transcript provided."

    llm = get_llm()

    chain = build_chain(
        llm=llm,
        system_prompt=(
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all action items. For each item, provide:\n"
            "- Task description\n"
            "- Owner (who is responsible)\n"
            "- Deadline (if mentioned, otherwise write 'Not specified')\n\n"
            "Format the result as a numbered list. Do not invent information. "
            "If none are found, say 'No action items found'."
        ),
    )

    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    """Extract important decisions from a meeting transcript."""

    if not transcript or not transcript.strip():
        return "No transcript provided."

    llm = get_llm()

    chain = build_chain(
        llm=llm,
        system_prompt=(
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract all important decisions made. Format them as a numbered "
            "list. Include context when useful. Do not invent information. "
            "If none are found, say 'No key decisions found'."
        ),
    )

    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    """Extract unanswered questions and follow-up topics."""

    if not transcript or not transcript.strip():
        return "No transcript provided."

    llm = get_llm()

    chain = build_chain(
        llm=llm,
        system_prompt=(
            "You are an expert meeting analyst. From the meeting transcript, "
            "extract unanswered questions and topics requiring follow-up. "
            "Format them as a numbered list. Do not invent information. "
            "If none are found, say 'No open questions found'."
        ),
    )

    return chain.invoke({"text": transcript})