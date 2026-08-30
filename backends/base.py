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


@runtime_checkable
class Backend(Protocol):
    """`generate(profile, prompt, model=...) -> str`.

    `profile` is compiled *inside* the call (`profile.system_prompt()`,
    `profile.options()`), never before it -- a backend with no equivalent of
    one of `options()`'s keys (Anthropic has no `repeat_penalty`) drops that
    key rather than the caller needing to know which backend it's talking to.
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
    print("ok")


if __name__ == "__main__":
    _self_check()
