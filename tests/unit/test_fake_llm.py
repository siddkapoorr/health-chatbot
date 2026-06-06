from health_intake.engine.extraction import FieldExtraction
from health_intake.llm.client import FakeLLMClient
from health_intake.models.conversation import ChatMessage


def test_fake_client_returns_scripted_extraction_and_reply() -> None:
    client = FakeLLMClient(
        extractions=[FieldExtraction(full_name="Jane Doe")],
        replies=["Thanks, Jane!"],
    )
    messages = [ChatMessage(role="user", content="I'm Jane Doe")]

    extraction = client.extract("system", messages)
    reply = client.generate("system", messages, "ask for DOB")

    assert extraction.full_name == "Jane Doe"
    assert reply == "Thanks, Jane!"
