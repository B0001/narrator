"""Anthropic backend: the opt-in `Backend` (see `base.py`), cloud, needs a key.

Second implementation of the same seam `ocean_ollama.py` was moved behind --
this is the backend `prd.md`'s "Decision: local-by-default, cloud opt-in"
(`narrator-c5b.1`) reverses the old "local Ollama only" non-goal for. Three
shapes differ from Ollama's `/api/generate`: `system` is a top-level request
field here, not folded into the prompt; `max_tokens` is required by
Anthropic and has no equivalent in `profile.options()`, so this module picks
a default; and a reply is a list of content blocks, not one `response`
string, so the text has to be pulled out of it.

`claude-fable-5` rejects `temperature` / `top_p` / `top_k` outright (HTTP
400 -- the model removed all sampler controls), so -- per `base.py`'s "drop
what the backend has no equivalent for" -- `profile.options()` is not
forwarded at all. On Fable the persona reaches the model through the system
prompt only; `prd.md` C1's trait-to-sampler path is inert on this backend.

The key comes from `ANTHROPIC_API_KEY` in the environment, never from source
or a profile file (prd.md, `narrator-c5b.1`). With no key set this fails by
naming that variable, before any network attempt -- not a connection error.

Server-side fallbacks are on by default (`anthropic-beta:
server-side-fallback-2026-07-01` header, `fallbacks: "default"` in the
body): a policy-decline gets one retry on a fallback model inside the same
call instead of surfacing straight to the caller as a refusal. Betas are an
HTTP header (`anthropic-beta`), not a JSON body field -- that's true for any
raw-HTTP request regardless of SDK; the SDK's `betas=[...]` kwarg is sugar
for the same header. This only softens *policy* declines; a chain that
refuses all the way through still raises below, same as before this bead.

`max_tokens` stays one flat default (`DEFAULT_MAX_TOKENS`) rather than a
per-caller value. It's a ceiling, not a spend -- billed output tokens track
what the model actually writes, not the cap -- and every current caller
(`motive.prose`, `chapters.write_chapters`, a single chat turn) calls
through `Backend.__call__(profile, prompt, model=...)`, which has no
`max_tokens` slot to plumb a per-caller value through in the first place
(`base.py`). Widening that protocol to carry one is a bigger, cross-cutting
change than this bead's remit (backends/base.py plus every caller's
signature); a shared ceiling generous enough for the longest current output
-- `chapters.write_chapters` prose, still one call per clue node over the
handful of fixed nodes `prd.md`'s cost table describes -- costs nothing extra
for the shorter calls, so one flat number is the right call until a caller
actually needs to differentiate.

    python3 backends/ocean_fable.py   # self-check (no key, no live call needed)
"""

import json
import os
import urllib.request

try:
    from .base import Reply  # imported as backends.ocean_fable (e.g. from ocean.py)
except ImportError:
    from base import Reply  # run directly: `python ocean_fable.py`, backends/ is on sys.path

DEFAULT_MAX_TOKENS = 4096
ANTHROPIC_VERSION = "2023-06-01"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def generate(profile, prompt, model="claude-fable-5", max_tokens=DEFAULT_MAX_TOKENS,
             host="https://api.anthropic.com"):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set -- the Fable backend needs it in the "
            "environment (never in source or a profile file)."
        )

    # No sampling params: claude-fable-5 400s on temperature/top_p/top_k, and
    # profile.options() carries exactly those. The system prompt does the work.
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": profile.system_prompt(),
        "messages": [{"role": "user", "content": prompt}],
        "fallbacks": "default",
    }).encode()
    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": FALLBACK_BETA,
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(f"{host}/v1/messages", body, headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.load(r)

    # A safety refusal is HTTP 200 with stop_reason "refusal" and usually no
    # text -- surface it rather than returning "" as if the model had answered.
    if reply.get("stop_reason") == "refusal":
        raise RuntimeError(
            "claude-fable-5 declined this request (stop_reason 'refusal'); "
            "nothing was generated."
        )
    text = "".join(block["text"] for block in reply["content"] if block.get("type") == "text")
    # The response names which model actually answered -- with server-side
    # fallbacks on, that can silently differ from the requested `model` (the
    # whole point of the fallback: a decline on the primary model re-runs on
    # another one inside the same call). narrator-c5b.2.4: a caller must be
    # able to see that swap, or an explicit "unknown", never assume the
    # requested name is who replied.
    return Reply(text, model=reply.get("model", model))


def _self_check():
    from base import conforms

    assert conforms(generate), "generate() must satisfy the Backend interface"

    class FakeProfile:
        def system_prompt(self):
            return "a system prompt"

        def options(self):
            # narrator-c5b.2.3: every trait's sampler target on this backend
            # is "none" (ocean.py's SAMPLER_TARGETS is Ollama-shaped and
            # never forwarded here) -- proved by raising if this is ever
            # called, not just by omitting "options" from the asserted body
            # below.
            raise AssertionError("Fable must never call profile.options() -- it forwards no sampler settings")

    # No key set: must fail naming the variable, without touching the network.
    real_urlopen = urllib.request.urlopen

    def exploding_urlopen(*a, **k):
        raise AssertionError("must not attempt a network call with no key set")

    urllib.request.urlopen = exploding_urlopen
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        try:
            generate(FakeProfile(), "a prompt")
        except RuntimeError as e:
            assert "ANTHROPIC_API_KEY" in str(e), "error must name the missing variable"
        else:
            raise AssertionError("must raise when ANTHROPIC_API_KEY is unset")
    finally:
        urllib.request.urlopen = real_urlopen
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key

    # Key set: request shape is right, response is parsed out of content blocks.
    captured = {}

    def fake_response(payload):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(payload).encode()

        return FakeResponse()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data)
        return captured["response"]

    urllib.request.urlopen = fake_urlopen
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        captured["response"] = fake_response({
            "stop_reason": "end_turn",
            "model": "test-model",
            "content": [
                {"type": "text", "text": "generated "},
                {"type": "text", "text": "text"},
            ],
        })
        result = generate(FakeProfile(), "a prompt", model="test-model")

        assert result == "generated text", "must join the 'text' fields of the content blocks"
        assert isinstance(result, Reply), "generate() must return a Reply, not a plain str"
        assert result.model == "test-model", "must report the model that answered"
        assert captured["url"] == "https://api.anthropic.com/v1/messages"
        assert captured["timeout"] == 120
        assert captured["headers"]["x-api-key"] == "test-key"
        assert captured["headers"]["anthropic-version"] == ANTHROPIC_VERSION
        assert captured["headers"]["anthropic-beta"] == FALLBACK_BETA, \
            "fallback beta must be sent as a header, not a body field"
        assert captured["body"] == {
            "model": "test-model",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": "a system prompt",
            "messages": [{"role": "user", "content": "a prompt"}],
            "fallbacks": "default",
        }, "request body: system top-level, no sampling params (Fable rejects them), fallbacks on by default"

        # A refusal must raise, not return "".
        captured["response"] = fake_response({"stop_reason": "refusal", "content": []})
        try:
            generate(FakeProfile(), "a prompt")
        except RuntimeError as e:
            assert "refusal" in str(e), "a refusal must be surfaced, got: " + str(e)
        else:
            raise AssertionError("stop_reason 'refusal' must raise")

        # Server-side fallback: the requested model declines and a fallback
        # model answers instead, inside the same call. The response's "model"
        # field is the one that actually replied, and a caller reading
        # .model must see that swap, not the name it originally requested.
        captured["response"] = fake_response({
            "stop_reason": "end_turn",
            "model": "claude-fallback-model",
            "content": [{"type": "text", "text": "a fallback reply"}],
        })
        swapped = generate(FakeProfile(), "a prompt", model="claude-fable-5")
        assert swapped.model == "claude-fallback-model", (
            "a fallback swap must be visible on .model, not silently reported as the requested model"
        )

        # If a response ever omits "model", fall back to what was requested
        # rather than crashing or guessing something else.
        captured["response"] = fake_response({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "no model field"}],
        })
        no_field = generate(FakeProfile(), "a prompt", model="requested-model")
        assert no_field.model == "requested-model", "with no 'model' in the response, fall back to what was requested"
    finally:
        urllib.request.urlopen = real_urlopen
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    print("ok")


if __name__ == "__main__":
    _self_check()
