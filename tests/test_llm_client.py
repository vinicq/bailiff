import instructor
import openai
import pytest

from bailiff.features.assistant.llm import LLMClient, LLMClientSettings


def _client():
    return LLMClient(LLMClientSettings(
        api_key="sk-test",
        base_url="http://localhost:11434/v1",
        model="llama3.2",
    ))


def test_llm_client_uses_raw_openai_for_freetext_chat():
    client = _client()
    assert isinstance(client._openai, openai.OpenAI)


def test_llm_client_exposes_instructor_wrapper_for_structured_output():
    client = _client()
    assert isinstance(client.structured, instructor.Instructor)


def test_chat_returns_string_when_provider_replies_with_content(mocker):
    client = _client()
    fake_choice = mocker.MagicMock()
    fake_choice.message.content = "PONG"
    fake_response = mocker.MagicMock()
    fake_response.choices = [fake_choice]
    mocker.patch.object(
        client._openai.chat.completions, "create", return_value=fake_response
    )

    reply = client.chat([{"role": "user", "content": "ping"}])
    assert reply == "PONG"


def test_chat_returns_sentinel_when_provider_replies_with_none(mocker):
    client = _client()
    fake_choice = mocker.MagicMock()
    fake_choice.message.content = None
    fake_response = mocker.MagicMock()
    fake_response.choices = [fake_choice]
    mocker.patch.object(
        client._openai.chat.completions, "create", return_value=fake_response
    )

    reply = client.chat([{"role": "user", "content": "ping"}])
    assert reply == "[no response]"


def test_chat_does_not_require_response_model_argument(mocker):
    client = _client()
    spy = mocker.patch.object(client._openai.chat.completions, "create")
    spy.return_value = mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(content="ok"))])

    client.chat([{"role": "user", "content": "hi"}])

    call_kwargs = spy.call_args.kwargs
    assert "response_model" not in call_kwargs
    assert call_kwargs["model"] == "llama3.2"
    assert call_kwargs["timeout"] == 30
