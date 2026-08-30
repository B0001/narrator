"""Backends that turn an Ocean profile + prompt into text.

Ollama (`ocean_ollama.py`) stays the default everywhere, local and keyless.
Fable (`ocean_fable.py`) is the opt-in second implementation, cloud and
keyed from `ANTHROPIC_API_KEY` -- see `base.py` for the interface both
satisfy.
"""
