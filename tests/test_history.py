from docwise.history import ConversationHistory


def test_streamlit_style_messages_restore_user_assistant_and_topic_metadata():
    messages = [
        {"role": "user", "content": "List the projects."},
        {
            "role": "assistant",
            "content": "1. Alpha\n2. Beta",
            "sources": ["ignored by history"],
            "resolved_question": "List all resume projects.",
        },
    ]

    history = ConversationHistory.from_messages(messages)

    assert [turn.role for turn in history.turns] == ["user", "assistant"]
    assert history.turns[1].content == "1. Alpha\n2. Beta"
    assert history.turns[1].resolved_question == "List all resume projects."
    assert "Assistant: 1. Alpha\n2. Beta" in history.formatted()


def test_ordinal_uses_latest_assistant_numbered_list_not_document_position():
    history = ConversationHistory()
    history.add_assistant("1. Alpha Project\n2. Beta Project\n3. Gamma Project")

    resolution = history.resolve_reference("What technologies were used in the 2nd project?")

    assert resolution.used_history is True
    assert resolution.anchor == "Beta Project"
    assert resolution.question == "What technologies were used in the Beta Project?"


def test_previous_example_uses_latest_resolved_conversation_topic():
    history = ConversationHistory()
    history.add_user("Explain the SharePoint filtering example.")
    history.add_assistant(
        "It filters choices based on department.",
        resolved_question="Explain the SharePoint Filtering Tool example.",
    )

    resolution = history.resolve_reference("What was the business value of the previous example?")

    assert resolution.used_history is True
    assert "Explain the SharePoint Filtering Tool example" in resolution.question
