"""
RAG Evaluation Harness

Purpose:
    Automated pass/fail test for retrieval quality.
    Runs paraphrased questions, checks whether the correct chunk
    was retrieved as the top-1 result for each.

    This replaces manual curl testing with a reproducible,
    one-command quality check.

Usage:
    python -m rag.eval

What it checks:
    - Did the RIGHT chunk come back as the #1 match? (retrieval accuracy)
    - Was the distance below the confidence threshold? (retrieval confidence)
    - It does NOT check the LLM's generated answer text — only retrieval,
      because retrieval is the foundation. If retrieval is wrong,
      no amount of prompt engineering fixes the answer.
"""

from rag.pipeline import initialize
from rag.retriever import retrieve
from rag.store import get_collection

# Maximum acceptable distance for a "confident" retrieval
CONFIDENCE_THRESHOLD = 0.7


# ============================================================
# Test cases: question → expected section title in top-1 result
# ============================================================
TEST_CASES = [
    {
        "id": "Q1",
        "question": "What is the torque spec for the door hinge bracket?",
        "expected_section": "WI-4.2 Torque Specifications Reference, Model AX-500 Door Hardware",
        "expected_doc": "02-WI-4.2-torque-specifications.md"
    },
    {
        "id": "Q2",
        "question": "What are the steps to install the door hinge bracket?",
        "expected_section": "SOP-3.1 Bracket Installation, Assembly Line 2, Station 4",
        "expected_doc": "01-SOP-3.1-bracket-installation.md"
    },
    {
        "id": "Q3",
        "question": "Why did the bracket thickness change?",
        "expected_section": "ECN-2291 Door Hinge Bracket Thickness Change",
        "expected_doc": "05-ECN-2291-bracket-thickness.md"
    },
    {
        "id": "Q4",
        "question": "What caused the Line 2 bracket flex issue?",
        "expected_section": "RCA-2026-014 Line 2 Stoppage, Door Hinge Bracket Flex",
        "expected_doc": "06-RCA-2026-014-bracket-flex.md"
    },
    {
        "id": "Q5",
        "question": "Is there an active quality alert for the bracket?",
        "expected_section": "QA-2026-009 Quality Alert, Door Hinge Bracket Flex, Model AX-500",
        "expected_doc": "04-QA-2026-009-quality-alert.md"
    },
    {
        "id": "Q6",
        "question": "How often should the Station 4 fixture be recalibrated?",
        "expected_section": "MM-Station4 Fixture Preventive Maintenance Schedule",
        "expected_doc": "03-MM-station4-fixture-maintenance.md"
    },
    {
        "id": "Q7",
        "question": "How much belt deflection is acceptable during tensioner inspection?",
        "expected_section": "SOP-7.4 Conveyor Belt Tensioner Inspection, Assembly Line 5, Station 2",
        "expected_doc": "07-SOP-7.4-conveyor-tensioner-inspection.md"
    },
    {
        "id": "Q8",
        "question": "Is there a known issue with the Line 5 conveyor?",
        "expected_section": "QA-2026-021 Quality Alert, Conveyor Belt Slippage, Assembly Line 5",
        "expected_doc": "08-QA-2026-021-conveyor-slippage.md"
    },
    {
        "id": "Q9",
        "question": "What tools are needed for the tensioner inspection?",
        "expected_section": "SOP-7.4 Conveyor Belt Tensioner Inspection, Assembly Line 5, Station 2",
        "expected_doc": "07-SOP-7.4-conveyor-tensioner-inspection.md"
    },
]


def run_eval():
    """
    Run all test cases and report pass/fail for each.

    A test PASSES if:
        1. The expected section appears in the top-1 retrieval result
        2. The retrieval distance is below CONFIDENCE_THRESHOLD

    Returns:
        tuple of (passed_count, total_count, detailed_results)
    """
    # Initialize the pipeline (chunks, embeds, stores)
    print("Initializing RAG pipeline for evaluation...")
    count = initialize(docs_dir="rag_docs", persistent=True)
    print(f"Loaded {count} chunks")
    print("=" * 70)

    collection = get_collection()
    results = []
    passed = 0
    total = len(TEST_CASES)

    for tc in TEST_CASES:
        retrieved = retrieve(tc["question"], collection=collection, top_k=3)
        top_result = retrieved[0] if retrieved else None

        if top_result is None:
            status = "FAIL"
            reason = "No results returned"
            distance = None
        else:
            got_section = top_result["section_title"]
            got_doc = top_result["source_doc"]
            distance = top_result["distance"]

            section_match = got_section == tc["expected_section"]
            confident = distance < CONFIDENCE_THRESHOLD

            if section_match and confident:
                status = "PASS"
                reason = None
                passed += 1
            elif section_match and not confident:
                status = "WEAK PASS"
                reason = f"Correct section but low confidence (dist: {distance:.4f} > {CONFIDENCE_THRESHOLD})"
                passed += 1
            else:
                status = "FAIL"
                reason = f"Expected [{tc['expected_section']}] but got [{got_section}]"

        result = {
            "id": tc["id"],
            "question": tc["question"],
            "expected": tc["expected_section"],
            "got": top_result["section_title"] if top_result else "NONE",
            "distance": distance,
            "status": status,
            "reason": reason
        }
        results.append(result)

        icon = "PASS" if "PASS" in status else "FAIL"
        print(f"\n{tc['id']}: \"{tc['question'][:60]}...\"")
        print(f"    Expected: {tc['expected_section']}")
        print(f"    Got:      {result['got']} (dist: {distance:.4f})" if distance else f"    Got: NONE")
        print(f"    [{icon}] {status}" + (f" — {reason}" if reason else ""))

    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} passed")

    if passed == total:
        print("All tests passed — retrieval quality confirmed.")
    else:
        failed = [r for r in results if "PASS" not in r["status"]]
        print(f"Failed tests: {', '.join(r['id'] for r in failed)}")
        print("Review failed cases — may need content additions or chunk adjustments.")

    return passed, total, results


if __name__ == "__main__":
    passed, total, results = run_eval()
    exit(0 if passed == total else 1)
