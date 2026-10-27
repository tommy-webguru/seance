#!/usr/bin/env python3
"""seance: hold a séance with your git history.

Every repository is haunted. The people who wrote it (including the you of
two years ago) left their reasons behind in commit messages, and nobody reads
them. seance lets you ask. A small open-weight model running on your own
machine (Gemma, through Ollama) reads the commits that match your question and
answers in the voice of the person who made them, citing the commits it leaned
on, and admitting when the record is silent.

    python seance.py                     # séance in the current repository
    python seance.py ~/code/old-project  # or any other one
    python seance.py --summon "Ana"      # talk to one spirit only
    python seance.py --haunt src/db.py   # whoever owns most of this file
    python seance.py --graveyard         # epitaphs for abandoned branches
    python seance.py --ofrenda "Ana"     # a remembrance for a past contributor

Standard library only. Needs git, and Ollama with a model pulled
(default: gemma3:4b). Nothing leaves your machine.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "gemma3:4b"
DEPARTED_AFTER = timedelta(days=180)   # no commits for this long: a spirit
STALE_BRANCH = timedelta(days=90)      # unmerged and untouched: a grave
TOP_K = 6                              # commits shown to the model per question
DIFF_CHARS = 700                       # diff excerpt per commit
STOP = {"the", "a", "an", "of", "to", "in", "on", "is", "it", "and", "or", "for",
        "why", "what", "how", "when", "who", "did", "do", "does", "was", "were",
        "you", "this", "that", "with", "from", "are", "be", "i", "we", "my", "your",
        "there", "here", "about", "into", "at", "by", "have", "has", "had", "not"}

_gpu_broken = False


# --------------------------------------------------------------------------- git

def git(repo, *args, check=True):
    proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if check and proc.returncode:
        raise SystemExit(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def load_commits(repo):
    """Every commit on every branch, newest first."""
    sep = "\x1e"
    fmt = "\x1d" + sep.join(["%h", "%H", "%an", "%ae", "%aI", "%s", "%b"]) + "\x1f"
    raw = git(repo, "log", "--all", "--no-merges", f"--format={fmt}", "--name-only")
    commits = []
    for chunk in raw.split("\x1d"):
        if not chunk.strip():
            continue
        head, _, files = chunk.partition("\x1f")
        parts = head.split(sep)
        if len(parts) < 7:
            continue
        short, full, name, email, date, subject, body = parts[:7]
        commits.append({
            "short": short, "hash": full, "author": name.strip(), "email": email.strip(),
            "date": datetime.fromisoformat(date), "subject": subject.strip(),
            "body": body.strip(), "files": [f for f in files.split("\n") if f.strip()],
        })
    return commits


def diff_excerpt(repo, commit, limit=DIFF_CHARS):
    text = git(repo, "show", commit["hash"], "--format=", "--unified=1", "--no-color",
               check=False)
    lines = [l for l in text.splitlines()
             if l.startswith(("+", "-", "@@", "diff --git")) and not l.startswith(("+++", "---"))]
    return "\n".join(lines)[:limit]


def spirits(commits, now):
    """Who haunts this repository, and who has moved on."""
    by = defaultdict(list)
    for c in commits:
        by[c["author"]].append(c)
    out = []
    for name, cs in by.items():
        first, last = min(c["date"] for c in cs), max(c["date"] for c in cs)
        out.append({"name": name, "commits": len(cs), "first": first, "last": last,
                    "departed": now - last > DEPARTED_AFTER})
    return sorted(out, key=lambda s: -s["commits"])


# --------------------------------------------------------------------- retrieval

def tokens(text):
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOP and len(t) > 1]


def relevant(commits, question, k=TOP_K):
    """Cheap keyword retrieval: enough for a 4B model to work from real evidence."""
    q = set(tokens(question))
    if not q:
        return commits[:k]
    scored = []
    for c in commits:
        hay = Counter(tokens(c["subject"]) * 3 + tokens(c["body"]) + tokens(" ".join(c["files"])) * 2)
        score = sum(hay[t] for t in q)
        # partial matches: "auth" should find "authentication"
        score += sum(0.5 for t in q for h in hay if t != h and (t in h or h in t) and len(t) > 3)
        if score:
            scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], -x[1]["date"].timestamp()))
    return [c for _, c in scored[:k]]


def describe(repo, c, with_diff=True):
    when = c["date"].strftime("%Y-%m-%d")
    files = ", ".join(c["files"][:8]) + (" ..." if len(c["files"]) > 8 else "")
    text = f"[{c['short']}] {when} by {c['author']}\n{c['subject']}"
    if c["body"]:
        text += "\n" + c["body"][:600]
    text += f"\nfiles: {files}"
    if with_diff:
        diff = diff_excerpt(repo, c)
        if diff:
            text += "\ndiff excerpt:\n" + diff
    return text


# ------------------------------------------------------------------------ ollama

def _post(body):
    """POST to Ollama's chat endpoint, falling back to the CPU if the GPU can't load the model."""
    global _gpu_broken
    while True:
        if _gpu_broken:
            body["options"]["num_gpu"] = 0
        req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            e.close()
            if e.code == 500 and not _gpu_broken:
                _gpu_broken = True
                print("  (the GPU would not take the spirit; trying the CPU)", file=sys.stderr)
                continue
            raise SystemExit(f"Ollama error {e.code}: {detail[:300]}")
        except urllib.error.URLError as e:
            raise SystemExit(f"Cannot reach Ollama at {OLLAMA} ({e.reason}). Is it running? "
                             f"Install it from https://ollama.com and run: ollama pull {MODEL}")


def ollama(messages, model, temperature=0.4):
    """Stream a chat completion from Ollama, piece by piece."""
    body = {"model": model, "messages": messages, "stream": True,
            "options": {"temperature": temperature, "num_ctx": 8192}}
    resp = _post(body)
    for line in resp:
        chunk = json.loads(line)
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece
        if chunk.get("done"):
            break


def check_model(model):
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"Cannot reach Ollama at {OLLAMA} ({getattr(e, 'reason', e)}). "
                         f"Install it from https://ollama.com and run: ollama pull {model}")
    if model not in names and f"{model}:latest" not in names:
        raise SystemExit(f"Model {model!r} is not pulled. Run: ollama pull {model}")


def whisper(stream, delay=0.012):
    """Print a streamed answer slowly enough to feel like it's arriving from somewhere."""
    out = []
    for piece in stream:
        for ch in piece:
            sys.stdout.write(ch)
            sys.stdout.flush()
            out.append(ch)
            if delay and ch not in "\n":
                time.sleep(delay)
    print()
    return "".join(out)


# ----------------------------------------------------------------------- prompts

GHOST_RULES = """You are a ghost: the spirit of {who}, speaking from beyond about a git repository you worked on.
Speak in the first person, past tense, with the weary candour of the dead and a little dry wit. Keep it short: a few sentences, at most two short paragraphs.
If the records come from more than one person, you are their chorus: say "we", and name who did what when it matters.
You remember ONLY what is in the commit records below. Every claim must come from them. Cite the commits you rely on by putting their short hash in square brackets, exactly as it appears at the start of each record. Valid hashes: {hashes}.
If the records do not answer the question, say plainly that this part of your memory has faded, and suggest what the living could look at instead. Never invent commits, files, dates or reasons.
Speak naturally; do not repeat the record headers or list the records back."""

ASK_TEMPLATE = """The living ask: {question}

Commit records (newest first):

{records}"""


def ghost_name(commits, summon=None):
    if summon:
        return summon
    authors = Counter(c["author"] for c in commits)
    if len(authors) == 1:
        return next(iter(authors))
    return "the people who wrote this repository (" + ", ".join(a for a, _ in authors.most_common(3)) + ")"


def answer(repo, commits, question, model, summon=None, show=False):
    pool = [c for c in commits if not summon or summon.lower() in c["author"].lower()]
    if not pool:
        print(f"  No commits by anyone called {summon!r}. The candles flicker; nobody comes.")
        return ""
    picks = relevant(pool, question)
    if len(picks) < TOP_K and (summon or len(pool) <= 3 * TOP_K):
        # a small circle: let the newest memories fill the empty chairs
        picks += [c for c in pool if c not in picks][:TOP_K - len(picks)]
    if not picks:
        print("  The spirits hear the question, but nothing in the history answers to those words.")
        return ""
    records = "\n\n".join(describe(repo, c) for c in picks)
    if show:
        print(textwrap.indent(records, "    "), end="\n\n")
    valid = [c["short"] for c in picks]
    messages = [{"role": "system", "content": GHOST_RULES.format(who=ghost_name(pool, summon), hashes=", ".join(valid))},
                {"role": "user", "content": ASK_TEMPLATE.format(question=question, records=records)}]
    print(f"  ({len(picks)} commit(s) stir: {', '.join(valid)})\n")
    return whisper(exorcise(ollama(messages, model), valid))


def exorcise(stream, valid):
    """Strip citations of commits that were never shown to the model. Ghosts embellish."""
    buf = ""
    for piece in stream:
        buf += piece
        while True:
            m = re.search(r"\[([0-9a-f]{6,40})\]", buf)
            if not m:
                break
            head, buf = buf[:m.end()], buf[m.end():]
            if not any(h.startswith(m.group(1)) or m.group(1).startswith(h) for h in valid):
                head = head[:m.start()] + "[a hash that was never in the records]"
            yield head
        if "[" not in buf:
            yield buf
            buf = ""
    if buf:
        yield buf


# ---------------------------------------------------------------------- features

def haunt(repo, commits, path):
    """Find who owns most of a file (git blame) and the commits that touched it."""
    blame = git(repo, "blame", "-w", "--line-porcelain", "--", path, check=False)
    owners = Counter(re.findall(r"^author (.+)$", blame, flags=re.M))
    if not owners:
        raise SystemExit(f"git blame knows nothing about {path!r}.")
    owner, lines = owners.most_common(1)[0]
    total = sum(owners.values())
    touched = [c for c in commits if any(f == path or f.endswith("/" + path) for f in c["files"])]
    print(f"  {path}: {total} lines, {lines} of them by {owner} "
          f"({100 * lines // max(total, 1)}%). {len(touched)} commit(s) touched it.")
    return owner, touched


def graveyard(repo, commits, model, now):
    """Unmerged branches nobody has touched in a while, each with a short epitaph."""
    raw = git(repo, "for-each-ref", "--format=%(refname:short)\x1f%(committerdate:iso-strict)\x1f%(objectname:short)",
              "refs/heads", "refs/remotes")
    head = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    merged = set(git(repo, "branch", "-a", "--merged", "HEAD", "--format=%(refname:short)").split())
    graves = []
    for line in raw.splitlines():
        name, date, short = line.split("\x1f")
        if name in merged or name == head or name.endswith("/HEAD"):
            continue
        last = datetime.fromisoformat(date)
        if now - last < STALE_BRANCH:
            continue
        log = git(repo, "log", "-5", "--format=%h %s", name, "--not", head, check=False).strip()
        graves.append((name, last, short, log or "(no commits of its own)"))
    if not graves:
        print("  No graves here. Every branch is either merged or still breathing.")
        return []
    print(f"  {len(graves)} unmerged branch(es) untouched for {STALE_BRANCH.days}+ days.\n")
    for name, last, short, log in sorted(graves, key=lambda g: g[1]):
        age = (now - last).days
        print(f"  ✝ {name}  (last commit {last:%Y-%m-%d}, {age} days ago)")
        messages = [
            {"role": "system", "content": "You write epitaphs for abandoned git branches: one line, at most 18 words, "
                                          "wry but kind, specific to the commits given. No hashes, no quotes. "
                                          "Never use the words valiant, attempt, journey, or march."},
            {"role": "user", "content": f"Branch {name!r}, dead {age} days. Its commits:\n{log}\n\nEpitaph:"},
        ]
        print("    ", end="")
        whisper(ollama(messages, model, temperature=0.9), delay=0.008)
        print()
    return graves


def ofrenda(repo, commits, who, model, now):
    """A remembrance for a contributor, in the spirit of Día de los Muertos."""
    theirs = [c for c in commits if who.lower() in c["author"].lower()]
    if not theirs:
        raise SystemExit(f"Nobody called {who!r} ever committed here.")
    name = theirs[0]["author"]
    first, last = theirs[-1], theirs[0]
    files = Counter(f for c in theirs for f in c["files"])
    hours = Counter(c["date"].hour for c in theirs)
    busiest_hour = hours.most_common(1)[0][0]
    span = (last["date"] - first["date"]).days
    print(f"  ✿ {name}: {len(theirs)} commit(s) over {span} day(s), "
          f"{first['date']:%Y-%m-%d} to {last['date']:%Y-%m-%d}.")
    print(f"    Most often at {busiest_hour:02d}:00. Touched {len(files)} file(s); "
          f"lived mostly in {', '.join(f for f, _ in files.most_common(3))}.")
    print(f"    First words: \"{first['subject']}\"")
    print(f"    Last words:  \"{last['subject']}\"")
    if now - last["date"] > DEPARTED_AFTER:
        print(f"    Gone from this repository for {(now - last['date']).days} days.")
    print()
    facts = "\n".join(f"[{c['short']}] {c['date']:%Y-%m-%d} {c['subject']}" for c in theirs[:40])
    messages = [
        {"role": "system", "content":
            "You write a short remembrance (an ofrenda) for a contributor to a software project, in the spirit of "
            "Día de los Muertos: warm, specific, a little playful, never saccharine. Three to five sentences. "
            "Use only the facts given. Mention what they cared about, judging from their commits. "
            "Do not invent anything about their life outside this repository."},
        {"role": "user", "content": f"Contributor: {name}\nActive {first['date']:%B %Y} to {last['date']:%B %Y}, "
                                    f"{len(theirs)} commits, usually around {busiest_hour:02d}:00.\n"
                                    f"Most touched: {', '.join(f for f, _ in files.most_common(5))}\n"
                                    f"Commits:\n{facts}\n\nRemembrance:"},
    ]
    print("    ", end="")
    text = whisper(ollama(messages, model, temperature=0.6))
    return text


# ------------------------------------------------------------------------- main

def banner(repo, sp, now):
    name = os.path.basename(os.path.abspath(repo))
    print(f"\n  🕯  The candles are lit in {name}.")
    gone = [s for s in sp if s["departed"]]
    here = [s for s in sp if not s["departed"]]
    print(f"  {len(sp)} spirit(s) linger here: {len(here)} still commit, {len(gone)} have departed.\n")
    for s in sp[:12]:
        mark = "✝" if s["departed"] else "•"
        print(f"   {mark} {s['name']:<28} {s['commits']:>5} commit(s)   {s['first']:%Y-%m-%d} to {s['last']:%Y-%m-%d}")
    if len(sp) > 12:
        print(f"     ... and {len(sp) - 12} more")
    print()


def main(argv=None):
    p = argparse.ArgumentParser(description="Hold a séance with your git history.")
    p.add_argument("repo", nargs="?", default=".", help="path to a git repository (default: current)")
    p.add_argument("--ask", metavar="QUESTION", help="ask one question and leave")
    p.add_argument("--summon", metavar="NAME", help="speak only with commits by this author")
    p.add_argument("--haunt", metavar="FILE", help="summon whoever wrote most of this file")
    p.add_argument("--graveyard", action="store_true", help="epitaphs for abandoned branches")
    p.add_argument("--ofrenda", metavar="NAME", help="a remembrance for a contributor")
    p.add_argument("--model", default=MODEL, help=f"Ollama model (default {MODEL})")
    p.add_argument("--show", action="store_true", help="print the commit records the ghost was given")
    p.add_argument("--fast", action="store_true", help="no typing effect")
    a = p.parse_args(argv)

    repo = os.path.abspath(a.repo)
    if not os.path.isdir(os.path.join(repo, ".git")) and git(repo, "rev-parse", "--git-dir", check=False).strip() == "":
        raise SystemExit(f"{repo!r} is not a git repository.")
    if a.fast:
        global whisper
        _w = whisper
        whisper = lambda stream, delay=0: _w(stream, 0)  # noqa: E731

    now = datetime.now(timezone.utc).astimezone()
    commits = load_commits(repo)
    if not commits:
        raise SystemExit("This repository has no commits. Nothing to summon.")
    check_model(a.model)

    if a.graveyard:
        graveyard(repo, commits, a.model, now)
        return 0
    if a.ofrenda:
        ofrenda(repo, commits, a.ofrenda, a.model, now)
        return 0

    summon = a.summon
    pool = commits
    if a.haunt:
        owner, touched = haunt(repo, commits, a.haunt)
        summon, pool = owner, (touched or commits)
        print(f"  Summoning {owner}...\n")

    sp = spirits(pool, now)
    if not a.ask:
        banner(repo, sp, now)
    if a.ask:
        answer(repo, pool, a.ask, a.model, summon, a.show)
        return 0

    print("  Ask your question. (Enter on an empty line, or Ctrl-C, ends the séance.)\n")
    try:
        while True:
            q = input("  you › ").strip()
            if not q:
                break
            print()
            answer(repo, pool, q, a.model, summon, a.show)
            print()
    except (KeyboardInterrupt, EOFError):
        print()
    print("  The candles go out.\n")
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    sys.exit(main())
