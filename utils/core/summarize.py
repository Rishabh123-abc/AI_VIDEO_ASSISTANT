import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# LLM CONFIGURATION
# ==========================================

def get_llm():
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


# ==========================================
# SPLIT TRANSCRIPT
# ==========================================

def split_transcript(transcript: str) -> list[str]:
    if not transcript or not transcript.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(transcript)


# ==========================================
# SUMMARIZE TRANSCRIPT
# ==========================================

def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        raise ValueError(
            "Transcript is empty. Nothing to summarize."
        )

    llm = get_llm()
    output_parser = StrOutputParser()

    # Step 1: Summarize each transcript chunk.
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize this portion of a meeting transcript concisely. "
            "Preserve important discussion points, decisions, action "
            "items, deadlines, and names when relevant. Do not invent "
            "information."
        ),
        ("human", "{transcript}"),
    ])

    map_chain = map_prompt | llm | output_parser

    chunks = split_transcript(transcript)

    if not chunks:
        raise ValueError("No transcript chunks were generated.")

    chunk_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Summarizing transcript part {index}/{len(chunks)}..."
        )

        # The prompt expects the key "transcript".
        result = map_chain.invoke({"transcript": chunk})

        if result and result.strip():
            chunk_summaries.append(result.strip())

    if not chunk_summaries:
        raise RuntimeError(
            "The model did not generate any partial summaries."
        )

    # Step 2: Combine partial summaries.
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine the partial "
            "summaries into one professional meeting summary. Use clear "
            "headings and bullet points. Include key discussion points, "
            "decisions, action items, owners, and deadlines when available. "
            "Do not invent information."
        ),
        ("human", "{text}"),
    ])

    combined_chain = combined_prompt | llm | output_parser

    # The combined prompt expects the key "text".
    final_summary = combined_chain.invoke({"text": combined})

    return final_summary.strip()


# ==========================================
# GENERATE MEETING TITLE
# ==========================================

def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate a short, professional title for this meeting. "
            "Use a maximum of 8 words. Return only the title, without "
            "quotation marks or explanation."
        ),
        ("human", "{text}"),
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    title = title_chain.invoke({
        "text": transcript[:2000]
    })

    return title.strip().strip('"').strip("'")