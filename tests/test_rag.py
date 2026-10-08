from langchain_core.documents import Document

from docwise.history import ConversationHistory
from docwise.models import RetrievedChunk
from docwise.rag import INSUFFICIENT_ANSWER, RAGService, validate_inline_citations
from docwise.retrieval import RetrievalOptions


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query, options):
        self.queries.append(query)
        return self.results


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return next(self.responses)


def test_grounded_answer_preserves_valid_and_removes_fake_citations(settings):
    source = RetrievedChunk(
        Document(
            page_content="Annual leave is twenty days.",
            metadata={"source": "policy.pdf", "page": 3},
        ),
        score=0.9,
    )
    service = RAGService(settings, FakeRetriever([source]), FakeLLM(["Twenty days [1], not thirty [9]."]))
    response = service.answer("What is the leave allowance?", options=RetrievalOptions())
    assert "[1]" in response.answer
    assert "[9]" not in response.answer
    assert response.sources[0].citation == "policy.pdf, page 3"


def test_low_relevance_returns_clear_insufficient_context(settings):
    source = RetrievedChunk(
        Document(page_content="Unrelated", metadata={"source": "x.txt", "page": 1}),
        score=0.01,
    )
    service = RAGService(settings, FakeRetriever([source]), FakeLLM([]))
    response = service.answer("Unknown topic", options=RetrievalOptions())
    assert response.answer == INSUFFICIENT_ANSWER
    assert response.insufficient_context is True


def test_history_aware_rewrite(settings):
    history = ConversationHistory()
    history.add_user("Tell me about the leave policy.")
    service = RAGService(settings, FakeRetriever([]), FakeLLM(["annual leave policy duration"]))
    response = service.answer("How long is it?", history=history, options=RetrievalOptions())
    assert response.rewritten_question == "annual leave policy duration"


def test_numbered_project_reference_cannot_drift_to_wrong_project(settings):
    history = ConversationHistory()
    history.add_user("List all the projects mentioned in my resume.")
    history.add_assistant(
        """1. SharePoint Filtering Tool
2. Power Apps Inventory Features
3. Data Migration and Cleaning Project
4. Daily Meter Data Collection Automation""",
        resolved_question="List all projects mentioned in the resume.",
    )
    llm = FakeLLM(
        ["Explain Meter Data Collection and Booking Record Migration technologies."]
    )
    service = RAGService(settings, FakeRetriever([]), llm)

    rewritten = service.rewrite_question(
        "Explain the second project in more detail. What technologies were used, "
        "and what business problem did it solve?",
        history,
    )

    assert "Power Apps Inventory Features" in rewritten
    assert "Meter Data" not in rewritten
    assert "technologies" in rewritten
    assert "business problem" in rewritten


def test_resolved_query_drives_retrieval_but_history_is_not_evidence(settings):
    history = ConversationHistory()
    history.add_user("List the projects.")
    history.add_assistant(
        "1. SharePoint Tool\n2. **Power Apps Inventory Features** — used blockchain.",
        resolved_question="List the resume projects.",
    )
    source = RetrievedChunk(
        Document(
            page_content="Built inventory features with Power Apps and SharePoint lists.",
            metadata={"source": "resume.pdf", "page": 2},
        ),
        score=0.92,
    )
    retriever = FakeRetriever([source])
    llm = FakeLLM(
        [
            "Explain the Power Apps Inventory Features project, including technologies and business problem.",
            "The project used Power Apps and SharePoint lists [1].",
        ]
    )
    service = RAGService(settings, retriever, llm)

    response = service.answer(
        "Explain the second project. What technologies did it use?",
        history=history,
        options=RetrievalOptions(),
    )

    assert retriever.queries == [
        "Explain the Power Apps Inventory Features project, including technologies and business problem."
    ]
    assert response.rewritten_question == retriever.queries[0]
    answer_prompt = llm.calls[1][0].content
    assert "Conversation history (navigation only, not evidence)" in answer_prompt
    assert "used blockchain" in answer_prompt
    assert "Document evidence (the only factual source)" in answer_prompt
    assert "Built inventory features with Power Apps and SharePoint lists" in answer_prompt
    assert response.answer == "The project used Power Apps and SharePoint lists [1]."


def test_pronoun_follow_up_uses_last_resolved_topic(settings):
    history = ConversationHistory()
    history.add_user("Explain the second project.")
    history.add_assistant(
        "Power Apps Inventory Features improved stock tracking.",
        resolved_question="Explain the Power Apps Inventory Features project.",
    )
    service = RAGService(
        settings,
        FakeRetriever([]),
        FakeLLM(["Explain the Power Apps Inventory Features project further."]),
    )
    rewritten = service.rewrite_question("Explain it further.", history)
    assert rewritten == "Explain the Power Apps Inventory Features project further."


def test_new_self_contained_topic_is_not_contaminated_by_history(settings):
    history = ConversationHistory()
    history.add_user("Explain the inventory project.")
    history.add_assistant(
        "It used Power Apps.",
        resolved_question="Explain the Power Apps Inventory Features project.",
    )
    llm = FakeLLM([])
    service = RAGService(settings, FakeRetriever([]), llm)
    question = "List the education qualifications in the resume."
    assert service.rewrite_question(question, history) == question
    assert llm.calls == []


def test_citation_validator():
    assert validate_inline_citations("Supported [1], invented [3].", 1) == "Supported [1], invented ."
