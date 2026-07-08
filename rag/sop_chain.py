"""
SOP Chain — auto-generates an updated SOP from a pasted spec/ECN change.

Design decision:
    Retrieval here serves a different purpose than in rag_query(): instead of
    answering a question, it pulls the EXISTING full procedure related to the
    change, so the LLM can merge the new change into the complete document
    rather than only outputting the changed portion.
"""

import os
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv

from rag.store import get_collection
from rag.retriever import retrieve

load_dotenv()

LLM_MODEL = "gpt-4o"

_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SOP_SYSTEM_PROMPT = (
    "You are drafting an updated Standard Operating Procedure (SOP) for a manufacturing shop floor. "
    "You will be given: (1) a short change description (e.g. an engineering change notice), and "
    "(2) existing related procedure documents retrieved from the knowledge base. "
    "Merge the change into a COMPLETE, updated procedure — carry over any unaffected steps from the "
    "existing documents exactly as they are, and apply the described change where relevant. "
    "Clearly mark which steps are new or changed. "
    "If the retrieved documents do not contain any clearly related existing procedure, "
    "say so explicitly and note that this would be a new SOP requiring careful review, "
    "rather than inventing steps not grounded in the provided context. "
    "Format the output as a numbered procedure, matching the style of the retrieved examples where possible."
)


def generate_sop(change_text: str, top_k: int = 3) -> Dict:
    """
    Generate an updated SOP draft from a pasted spec/ECN change.

    Args:
        change_text: the user-pasted engineering change notice or spec change
        top_k: number of related existing documents to retrieve as context

    Returns:
        dict with:
            - draft: the generated SOP text
            - reference_docs: list of {document, section, distance} used as grounding
            - grounded: bool, whether any relevant existing documents were found
    """
    try:
        collection = get_collection()
    except Exception:
        return {
            "draft": "The knowledge base has not been initialized. Please restart the service.",
            "reference_docs": [],
            "grounded": False
        }

    # Retrieve existing related procedure(s) using the change text as the query
    retrieved = retrieve(change_text, collection=collection, top_k=top_k)

    CONFIDENCE_THRESHOLD = 0.5
    grounded = bool(retrieved and retrieved[0]["distance"] <= CONFIDENCE_THRESHOLD)

    reference_docs = [
        {
            "document": chunk["source_doc"],
            "section": chunk["section_title"],
            "distance": chunk["distance"]
        }
        for chunk in retrieved
    ]

    if grounded:
        context_parts = [
            f"[Source: {chunk['source_doc']} > {chunk['section_title']}]\n{chunk['text']}"
            for chunk in retrieved
        ]
        context_block = "\n\n---\n\n".join(context_parts)
        user_prompt = (
            f"Change description:\n{change_text}\n\n"
            f"Existing related documents:\n{context_block}\n\n"
            f"Draft the complete, updated SOP incorporating this change."
        )
    else:
        # No strong match — still attempt a draft, but instruct the model to flag it clearly
        user_prompt = (
            f"Change description:\n{change_text}\n\n"
            f"No closely related existing procedure was found in the knowledge base. "
            f"Draft a proposed new SOP based only on the change description, "
            f"and clearly flag that this is a new procedure requiring full review "
            f"since no existing baseline was found."
        )

    response = _openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SOP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=600,
        temperature=0.0
    )

    draft = response.choices[0].message.content

    return {
        "draft": draft,
        "reference_docs": reference_docs,
        "grounded": grounded
    }
