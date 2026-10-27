# 🕯 seance

**Hold a séance with your git history. Ask the people who wrote your code why, even if one of them is you, three years ago.**

Every repository is haunted. The reasons behind the code are all there in the commit messages, and nobody reads them. seance finds the commits that match your question and hands them to a small open-weight model running on your own machine (Gemma 3, through [Ollama](https://ollama.com)). The model answers in the voice of whoever made those commits, cites them by hash, and says so when the record is silent.

```
$ python seance.py ~/code/Deep-Live-Cam --ask "Why did you add the face enhancer, and what did it cost?"

  (6 commit(s) stir: e57116d, 9525d45, 4d04e83, cfa8123, 65da3be, a6c9960)

It was a mess, really. Initially, I added the GPEN face enhancers, the 256 and
512 models [e57116d], as a feature. They were based on ONNX models,
automatically downloaded from GitHub releases, and integrated into the existing
frame processor pipeline. ...
```

Nothing leaves your machine. The only network call is to `localhost:11434`.

## What it does

| Command | What happens |
|---|---|
| `python seance.py` | Lists who haunts the current repository (and who has departed), then takes questions. |
| `python seance.py --ask "..."` | One question, one answer, exit. |
| `python seance.py --summon "Ana"` | Only Ana's commits are allowed to speak. |
| `python seance.py --haunt src/db.py` | Finds who wrote most of that file (`git blame`) and summons them, with only the commits that touched it. |
| `python seance.py --graveyard` | Unmerged branches nobody has touched for 90 days, each with a one-line epitaph. |
| `python seance.py --ofrenda "Ana"` | A remembrance for a contributor: when they were here, what they cared about, their first and last words, in the spirit of Día de los Muertos. |

Add `--show` to see exactly which commit records the ghost was given. Add `--fast` to skip the typing effect.

## How it works

1. `git log --all` is parsed into commits: author, date, subject, body, files touched.
2. Your question is tokenised and scored against every commit (subject counts triple, file paths double, partial word matches half). No embeddings, no vector database: a few hundred commits score in milliseconds and the result is easy to inspect.
3. The top six commits are described to the model, each with a short diff excerpt.
4. Gemma answers under strict instructions: first person, only what the records say, cite hashes, admit when memory has faded.

The hard rule is in step 4. A 4B model will cheerfully invent history if you let it. Giving it the real commits and forbidding anything else turns a storyteller into a witness.

## Install

```bash
# Ollama from https://ollama.com, then:
ollama pull gemma3:4b          # ~3.3 GB; gemma3:1b on small machines
python seance.py               # inside any git repository
```

One Python file, standard library only. Needs `git` on the PATH. Any model Ollama can run works with `--model`.

If the GPU can't load the model (old drivers, too little VRAM), seance says so once and continues on the CPU. Expect 20 to 120 seconds per answer on a CPU, depending on how much diff text the chosen commits carry.

## Tests

```bash
python -m unittest -v test_seance
```

The tests build a throwaway repository and a fake Ollama, so they need neither a model nor a GPU and run in a few seconds.

## Limits

- Retrieval is keyword-based. Ask with the words people would have used in commit messages ("cache", "timeout", "login"), not abstractions ("the architecture").
- The ghost knows only commit messages, file lists and short diff excerpts. It has not read your whole codebase.
- Everything is as honest as your commit messages. "fix stuff" summons a very vague spirit.

## License

MIT.
