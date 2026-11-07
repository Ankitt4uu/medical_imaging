"""SQLite store: results, tamper-evident append-only audit log (triggers + hash chain)."""
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS results(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  orthanc_id TEXT UNIQUE NOT NULL,
  kind TEXT NOT NULL,                  -- classification | segmentation
  modality TEXT, patient_id TEXT, study_uid TEXT,
  status TEXT NOT NULL,                -- pending_review | approved | rejected | failed
  prediction TEXT, artifacts TEXT,
  model_version TEXT, demo_mode INTEGER DEFAULT 0, error TEXT,
  created_at TEXT NOT NULL,
  decided_by TEXT, decided_at TEXT, decision_comment TEXT
);
CREATE TABLE IF NOT EXISTS audit_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, user TEXT NOT NULL, action TEXT NOT NULL,
  result_id INTEGER, model_version TEXT, prediction TEXT, detail TEXT,
  prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
  BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
  BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TABLE IF NOT EXISTS worker_state(key TEXT PRIMARY KEY, value TEXT);
"""
GENESIS = "0" * 64


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    # ---- results
    def create_result(self, **f):
        cols = ["orthanc_id", "kind", "modality", "patient_id", "study_uid", "status", "prediction",
                "artifacts", "model_version", "demo_mode", "error"]
        vals = [json.dumps(f.get(c)) if c in ("prediction", "artifacts") and f.get(c) is not None else f.get(c) for c in cols]
        with self.lock:
            cur = self.conn.execute(
                f"INSERT INTO results({','.join(cols)},created_at) VALUES({','.join('?' * len(cols))},?)", vals + [now()])
            self.conn.commit()
            return cur.lastrowid

    def get_result(self, rid):
        with self.lock:
            r = self.conn.execute("SELECT * FROM results WHERE id=?", (rid,)).fetchone()
        return self._row(r) if r else None

    def has_orthanc_id(self, oid):
        with self.lock:
            return self.conn.execute("SELECT 1 FROM results WHERE orthanc_id=?", (oid,)).fetchone() is not None

    def list_results(self, status=None):
        q, a = "SELECT * FROM results", ()
        if status:
            q, a = q + " WHERE status=?", (status,)
        with self.lock:
            rows = self.conn.execute(q + " ORDER BY id DESC", a).fetchall()
        return [self._row(r) for r in rows]

    def decide(self, rid, decision, user, comment):
        """Atomic: only a pending_review result can be decided, exactly once."""
        status = "approved" if decision == "approve" else "rejected"
        with self.lock:
            cur = self.conn.execute(
                "UPDATE results SET status=?, decided_by=?, decided_at=?, decision_comment=? "
                "WHERE id=? AND status='pending_review'", (status, user, now(), comment, rid))
            self.conn.commit()
            return cur.rowcount == 1

    @staticmethod
    def _row(r):
        d = dict(r)
        for k in ("prediction", "artifacts"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        return d

    # ---- audit
    @staticmethod
    def _hash(prev, ts, user, action, rid, mv, pred, detail):
        payload = json.dumps([prev, ts, user, action, rid, mv, pred, detail], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def audit(self, user, action, result_id=None, model_version=None, prediction=None, detail=None):
        pred = json.dumps(prediction, sort_keys=True) if prediction is not None else None
        with self.lock:
            last = self.conn.execute("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
            prev, ts = (last[0] if last else GENESIS), now()
            h = self._hash(prev, ts, user, action, result_id, model_version, pred, detail)
            self.conn.execute(
                "INSERT INTO audit_log(ts,user,action,result_id,model_version,prediction,detail,prev_hash,entry_hash) "
                "VALUES(?,?,?,?,?,?,?,?,?)", (ts, user, action, result_id, model_version, pred, detail, prev, h))
            self.conn.commit()

    def audit_entries(self, limit=500, result_id=None):
        q, a = "SELECT * FROM audit_log", ()
        if result_id is not None:
            q, a = q + " WHERE result_id=?", (result_id,)
        with self.lock:
            return [dict(r) for r in self.conn.execute(q + " ORDER BY id DESC LIMIT ?", a + (limit,))]

    def verify_audit_chain(self):
        with self.lock:
            rows = self.conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        prev = GENESIS
        for r in rows:
            h = self._hash(prev, r["ts"], r["user"], r["action"], r["result_id"], r["model_version"], r["prediction"], r["detail"])
            if r["prev_hash"] != prev or r["entry_hash"] != h:
                return {"valid": False, "broken_at": r["id"], "entries": len(rows)}
            prev = r["entry_hash"]
        return {"valid": True, "entries": len(rows)}

    # ---- worker state
    def get_state(self, key, default=None):
        with self.lock:
            r = self.conn.execute("SELECT value FROM worker_state WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_state(self, key, value):
        with self.lock:
            self.conn.execute("INSERT INTO worker_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self.conn.commit()
