"""Tests for seance. A fake Ollama and a throwaway git repository; no model needed.

Run: python -m unittest -v test_seance
"""

import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import seance


class FakeOllama(BaseHTTPRequestHandler):
    calls = []
    gpu_broken = False

    def log_message(self, *args):
        pass

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"models": [{"name": "gemma3:4b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.calls.append(body)
        if FakeOllama.gpu_broken and body["options"].get("num_gpu") != 0:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"cuda")
            return
        prompt = body["messages"][-1]["content"]
        hashes = [h for h in seance.re.findall(r"\[([0-9a-f]{7,})\]", prompt)]
        text = "I remember " + (f"[{hashes[0]}]" if hashes else "nothing") + ". Let me rest."
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for word in text.split(" "):
            self.wfile.write(json.dumps({"message": {"content": word + " "}, "done": False}).encode() + b"\n")
        self.wfile.write(json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n")


def run(repo, *args, env=None):
    e = dict(os.environ, **(env or {}))
    subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, env=e)


def commit(repo, msg, files, author, when):
    for name, text in files.items():
        path = os.path.join(repo, name)
        os.makedirs(os.path.dirname(path) or repo, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    run(repo, "add", "-A")
    stamp = when.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    run(repo, "commit", "-q", "-m", msg, env={
        "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": f"{author.lower()}@example.com",
        "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": f"{author.lower()}@example.com",
        "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})


class SeanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        seance.OLLAMA = f"http://127.0.0.1:{cls.server.server_port}"
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = cls.tmp.name
        run(cls.repo, "init", "-q", "-b", "main")
        run(cls.repo, "config", "user.name", "x")
        run(cls.repo, "config", "user.email", "x@example.com")
        long_ago = datetime(2024, 3, 1, 23, 0, tzinfo=timezone.utc)
        commit(cls.repo, "Add login form\n\nUsers kept asking for it.", {"auth/login.py": "def login(): pass\n"},
               "Ana", long_ago)
        commit(cls.repo, "Fix password reset token expiry\n\nTokens never expired. Oops.",
               {"auth/login.py": "def login(): pass\ndef reset(): pass\n"}, "Ana", long_ago + timedelta(days=3))
        commit(cls.repo, "Add README", {"README.md": "# demo\n"}, "Ben", datetime.now(timezone.utc) - timedelta(days=2))
        run(cls.repo, "checkout", "-q", "-b", "old-idea")
        commit(cls.repo, "Try a new cache layer", {"cache.py": "x = 1\n"}, "Ben", long_ago + timedelta(days=10))
        run(cls.repo, "checkout", "-q", "main")
        cls.commits = seance.load_commits(cls.repo)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def setUp(self):
        FakeOllama.calls = []
        FakeOllama.gpu_broken = False
        seance._gpu_broken = False

    def test_loads_every_branch_with_bodies_and_files(self):
        subjects = {c["subject"] for c in self.commits}
        self.assertEqual(subjects, {"Add login form", "Fix password reset token expiry", "Add README",
                                    "Try a new cache layer"})
        login = next(c for c in self.commits if c["subject"] == "Add login form")
        self.assertEqual(login["body"], "Users kept asking for it.")
        self.assertEqual(login["files"], ["auth/login.py"])
        self.assertEqual(login["author"], "Ana")

    def test_spirits_tell_departed_from_present(self):
        now = datetime.now(timezone.utc)
        by = {s["name"]: s for s in seance.spirits(self.commits, now)}
        self.assertTrue(by["Ana"]["departed"])
        self.assertFalse(by["Ben"]["departed"])
        self.assertEqual(by["Ana"]["commits"], 2)

    def test_retrieval_prefers_matching_commits_and_partial_words(self):
        picks = seance.relevant(self.commits, "why do password tokens expire?")
        self.assertEqual(picks[0]["subject"], "Fix password reset token expiry")
        picks = seance.relevant(self.commits, "anything about auth?")
        self.assertTrue(all("auth/login.py" in c["files"] for c in picks[:2]))
        self.assertEqual(seance.relevant(self.commits, "quantum teleportation"), [])

    def test_answer_cites_a_real_commit_and_sends_only_the_evidence(self):
        with redirect_stdout(io.StringIO()) as out:
            text = seance.answer(self.repo, self.commits, "why was the login form added?", "gemma3:4b")
        short = next(c["short"] for c in self.commits if c["subject"] == "Add login form")
        self.assertIn(f"[{short}]", text)
        prompt = FakeOllama.calls[-1]["messages"][-1]["content"]
        self.assertIn("Users kept asking for it.", prompt)
        self.assertIn("diff excerpt", prompt)
        self.assertIn("+def login", prompt)
        self.assertIn("commit(s) stir", out.getvalue())

    def test_summon_limits_to_one_author(self):
        with redirect_stdout(io.StringIO()) as out:
            seance.answer(self.repo, self.commits, "tell me about the README", "gemma3:4b", summon="Ana")
        self.assertIn("nothing in the history answers", out.getvalue())
        self.assertEqual(FakeOllama.calls, [])
        with redirect_stdout(io.StringIO()) as out:
            seance.answer(self.repo, self.commits, "anything", "gemma3:4b", summon="Nobody")
        self.assertIn("nobody comes", out.getvalue())
        with redirect_stdout(io.StringIO()):
            seance.answer(self.repo, self.commits, "login", "gemma3:4b", summon="Ana")
        self.assertNotIn("Add README", FakeOllama.calls[-1]["messages"][-1]["content"])

    def test_haunt_finds_the_owner_of_a_file(self):
        with redirect_stdout(io.StringIO()) as out:
            owner, touched = seance.haunt(self.repo, self.commits, "auth/login.py")
        self.assertEqual(owner, "Ana")
        self.assertEqual(len(touched), 2)
        self.assertIn("100%", out.getvalue())

    def test_graveyard_lists_only_stale_unmerged_branches(self):
        now = datetime.now(timezone.utc)
        with redirect_stdout(io.StringIO()) as out:
            graves = seance.graveyard(self.repo, self.commits, "gemma3:4b", now)
        self.assertEqual([g[0] for g in graves], ["old-idea"])
        self.assertIn("Try a new cache layer", FakeOllama.calls[-1]["messages"][-1]["content"])
        self.assertIn("old-idea", out.getvalue())

    def test_ofrenda_uses_only_that_persons_commits(self):
        now = datetime.now(timezone.utc)
        with redirect_stdout(io.StringIO()) as out:
            seance.ofrenda(self.repo, self.commits, "ana", "gemma3:4b", now)
        prompt = FakeOllama.calls[-1]["messages"][-1]["content"]
        self.assertIn("Add login form", prompt)
        self.assertNotIn("Add README", prompt)
        self.assertIn("First words", out.getvalue())
        self.assertIn("Gone from this repository", out.getvalue())

    def test_falls_back_to_cpu_when_the_gpu_refuses(self):
        FakeOllama.gpu_broken = True
        with redirect_stdout(io.StringIO()):
            text = seance.answer(self.repo, self.commits, "login form?", "gemma3:4b")
        self.assertIn("I remember", text)
        self.assertEqual(FakeOllama.calls[-1]["options"]["num_gpu"], 0)

    def test_cli_one_shot(self):
        with redirect_stdout(io.StringIO()) as out:
            code = seance.main([self.repo, "--ask", "password reset?", "--fast"])
        self.assertEqual(code, 0)
        self.assertIn("I remember", out.getvalue())


if __name__ == "__main__":
    unittest.main()
