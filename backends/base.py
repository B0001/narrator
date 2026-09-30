"""The interface every generation backend satisfies.

`ocean.generate()` used to POST straight to a local Ollama server -- the HTTP
call, the request shape, and the personality compilation were one function.
That made Ollama the only thing `agents.converse`, `turn.run_turn`,
`chapters.write_chapters`, and `motive.prose` could ever call, because
nothing about *how* text gets made was separated from *that* it gets made.

`Backend` is that separation, written down: take a profile (anything with
`system_prompt()` and `options()` -- an `Ocean`, or the fixed
`REASONING_PROFILE` turn.py pins reasoning calls to), a prompt, and a model
name; return text. Everything backend-specific -- the request shape, the
transport, auth, which sampler knobs exist at all -- lives inside the
function that implements this, not in the callers above it. That is the seam
a second backend (`narrator-c5b.2.2`) drops into without touching a caller:
swap which module's `generate` a caller imports (or passes as
`generate_fn`), and the interface is unchanged.

    python3 backends/base.py   # self-check
"""

from typing import Any, Protocol, runtime_checkable


class Reply(str):
    """A backend's response text, plus which model actually answered.

    Subclasses `str` so every existing caller (`.strip()`, `json.loads()`,
    equality, concatenation, f-strings) keeps working unmodified -- `Backend`
    still returns something that *is* a string, not something callers must
    unwrap. The `.model` attribute is additive: a caller that wants to record
    which model produced a reply (narrator-c5b.2.4 -- a transcript should
    name the model that answered, or an explicit "unknown", never silently
    assume one) reads `.model`; a caller that doesn't care never sees it.

    `getattr(x, "model", "unknown")` is the compatibility shim for code paths
    that may receive either a `Reply` or a plain `str` (e.g. a self-check's
    `fake_generate` stub, which has no model to report) -- it degrades to
    "unknown" rather than raising on the plain-`str` case.
    """

    def __new__(cls, text, model="unknown"):
        self = super().__new__(cls, text)
        self.model = model
        return self


@runtime_checkable
class Backend(Protocol):
    """`generate(profile, prompt, model=...) -> str`.

    `profile` is compiled *inside* the call (`profile.system_prompt()`,
    `profile.options()`), never before it -- a backend with no equivalent of
    one of `options()`'s keys (Anthropic has no `repeat_penalty`) drops that
    key rather than the caller needing to know which backend it's talking to.

    The return value is `str`-typed here because `Reply` (above) *is* a
    `str` -- a backend may return either a plain string or a `Reply` and
    still conform.
    """

    def __call__(self, profile: Any, prompt: str, model: str = ...) -> str:
        ...


def conforms(candidate) -> bool:
    """True if `candidate` is callable the way a `Backend` must be.

    A plain function already satisfies `Backend` structurally (it has
    `__call__`), so this is a readability aid for self-checks, not a gate
    anything import-time relies on -- Python has no way to check the
    parameter names or the return type without calling it.
    """
    return isinstance(candidate, Backend) and callable(candidate)


def _self_check():
    def good(profile, prompt, model="m"):
        return f"{model}:{prompt}"

    assert conforms(good), "a function taking (profile, prompt, model=...) must conform"
    assert not conforms(object()), "a non-callable must not conform"
    assert not conforms(5), "a non-callable must not conform"

    # Reply is a drop-in str: every operation a caller already does on a
    # backend's return value must work unchanged, plus the new .model.
    r = Reply("  generated text  ", model="claude-fable-5")
    assert r == "  generated text  ", "Reply must compare equal to the plain string"
    assert r.strip() == "generated text", "Reply must support str methods like a plain string"
    assert isinstance(r.strip(), str) and not isinstance(r.strip(), Reply), (
        "a Reply's str methods return plain str -- .model does not survive .strip(), "
        "so callers that want it must read it before transforming the text"
    )
    assert r.model == "claude-fable-5"
    assert f"reply was: {r}" == "reply was:   generated text  ", "Reply must interpolate like a str"
    assert isinstance(r, str), "Reply must satisfy isinstance(x, str) for any caller that checks"

    # A backend that returns a Reply still conforms to Backend.
    def good_reply(profile, prompt, model="m"):
        return Reply(f"{model}:{prompt}", model=model)

    assert conforms(good_reply)

    # No model given: falls back to "unknown", not a crash or a guessed name.
    assert Reply("text").model == "unknown"

    # getattr(x, "model", "unknown") is the shim callers use for "either a
    # Reply or a plain str" -- must degrade cleanly on a plain str.
    assert getattr("plain string", "model", "unknown") == "unknown"
    assert getattr(Reply("x", model="m"), "model", "unknown") == "m"

    print("ok")


if __name__ == "__main__":
    _self_check()
