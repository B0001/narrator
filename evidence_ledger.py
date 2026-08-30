"""Append-only evidence ledger with provenance tags (C5).

The C4 mystery kept ground truth (`WorldSimulation`) strictly apart from what
an investigation could observe (`EpistemicClueGraph`). C5 turns the chatbot
into that investigation: the user holds ground truth, and the only thing the
bot may reason from is what has actually been written to the ledger.

Provenance is the whole point. A chatbot that treats "the user told me X"
and "I inferred X" as the same kind of fact will state the second as if it
were the first, which is exactly the fair-play violation this repo exists to
catch. Four tags, two families:

  DIRECT  -- the user has actually seen this: they said it (`stated_by_user`)
             or an artifact showed it to them (`observed_artifact`).
  DERIVED -- the bot produced this itself: a deduction from other entries
             (`inferred_by_model`) or an unverified premise (`assumed`).

Only DIRECT entries are evidence a user has seen with their own eyes. A
DERIVED entry can still be admissible later (that is `admissibility.py`'s
job), but never because of its own provenance tag alone.

    python3 evidence_ledger.py   # self-check
"""

import json
from dataclasses import asdict, dataclass, field

DIRECT = frozenset({"stated_by_user", "observed_artifact"})
DERIVED = frozenset({"inferred_by_model", "assumed"})
PROVENANCE_TAGS = DIRECT | DERIVED


@dataclass
class LedgerEntry:
    id: str
    turn: int
    claim: str
    provenance: str
    supports: tuple = field(default_factory=tuple)  # ids this entry was derived from


class EvidenceLedger:
    """One append-only JSONL ledger per conversation.

    Every write is flushed immediately: a run killed mid-write leaves a
    truncated last line, never a lost-but-silent one, and `load` names the
    exact line that did not make it.
    """

    def __init__(self, path):
        self.path = path
        self._entries = {}
        self._order = []
        self._file = open(path, "a")

    def write(self, entry_id, turn, claim, provenance, supports=()):
        """Append one entry. Rejects reuse of an id and any untagged/unknown
        provenance at write time -- there is no such thing as evidence with
        no source, so refusing it here beats discovering it later at read time.
        """
        if entry_id in self._entries:
            raise ValueError(f"duplicate ledger entry id {entry_id!r}")
        if provenance not in PROVENANCE_TAGS:
            raise ValueError(
                f"entry {entry_id!r} has no valid provenance tag (got {provenance!r}); "
                f"must be one of {sorted(PROVENANCE_TAGS)}"
            )
        unknown_supports = [s for s in supports if s not in self._entries]
        if unknown_supports:
            raise ValueError(f"entry {entry_id!r} supports unknown ids {unknown_supports}")

        entry = LedgerEntry(entry_id, turn, claim, provenance, tuple(supports))
        self._entries[entry_id] = entry
        self._order.append(entry_id)
        self._file.write(json.dumps(asdict(entry)) + "\n")
        self._file.flush()
        return entry

    def get(self, entry_id):
        return self._entries[entry_id]

    def __contains__(self, entry_id):
        return entry_id in self._entries

    def entries(self):
        return [self._entries[eid] for eid in self._order]

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load(path):
    """Read a ledger back, naming the file and line if one will not parse.

    Mirrors agents.load_transcript: a flushed-per-line file killed mid-write
    leaves a half-written final object, and a bare JSONDecodeError reports a
    column inside a string it never shows you. The file and line number are
    what make the damage findable.
    """
    entries = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
            entries.append(LedgerEntry(
                raw["id"], raw["turn"], raw["claim"], raw["provenance"], tuple(raw["supports"]),
            ))
    return entries


def _self_check():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/ledger.jsonl"

        with EvidenceLedger(path) as ledger:
            ledger.write("e1", 0, "user said the window was locked", "stated_by_user")
            ledger.write("e2", 0, "photo shows the window latch engaged", "observed_artifact")
            ledger.write("e3", 1, "the intruder didn't come through the window", "inferred_by_model", supports=("e1", "e2"))

            # Untagged / unknown provenance is rejected at write time, not read time.
            try:
                ledger.write("bad", 1, "no source for this", "vibes")
            except ValueError as e:
                assert "provenance" in str(e)
            else:
                raise AssertionError("unknown provenance tag should have been rejected")

            # Duplicate ids are rejected.
            try:
                ledger.write("e1", 2, "restating", "stated_by_user")
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate id should have been rejected")

            # Supporting an entry that doesn't exist yet is rejected.
            try:
                ledger.write("e4", 2, "dangling support", "inferred_by_model", supports=("nope",))
            except ValueError:
                pass
            else:
                raise AssertionError("support referencing an unknown id should have been rejected")

        loaded = load(path)
        assert [e.id for e in loaded] == ["e1", "e2", "e3"], "ledger must read back in write order"
        assert loaded[2].supports == ("e1", "e2")
        assert loaded[0].provenance in DIRECT
        assert loaded[2].provenance in DERIVED

        # A run killed mid-write: the first two lines are whole, the third is
        # a truncated fragment.
        broken = f"{d}/broken.jsonl"
        good1 = json.dumps(asdict(LedgerEntry("e1", 0, "x", "stated_by_user", ())))
        good2 = json.dumps(asdict(LedgerEntry("e2", 0, "y", "stated_by_user", ())))
        with open(broken, "w") as f:
            f.write(good1 + "\n\n" + good2 + "\n" + '{"id": "e3", "turn": 1, "claim": "cut off mid')
        try:
            load(broken)
        except ValueError as e:
            # Blank line 2 is skipped and must not shift the count -- the
            # truncated object really is on line 4.
            assert f"{broken}:4:" in str(e), f"error must name the file and line, got {e}"
        else:
            raise AssertionError("a truncated final line should have been rejected")

        # A blank trailing line is normal, not corruption.
        with open(broken, "w") as f:
            f.write(good1 + "\n\n")
        assert len(load(broken)) == 1, "blank lines are skipped, not parsed"

        os.remove(broken)

    print("ok")


if __name__ == "__main__":
    _self_check()
