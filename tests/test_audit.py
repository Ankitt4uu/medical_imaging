import sqlite3

import pytest

from backend.app.db import DB


def test_audit_is_append_only():
    db = DB(":memory:")
    db.audit("alice", "login")
    with pytest.raises(sqlite3.DatabaseError):
        db.conn.execute("UPDATE audit_log SET user='mallory'")
    with pytest.raises(sqlite3.DatabaseError):
        db.conn.execute("DELETE FROM audit_log")
    assert len(db.audit_entries()) == 1


def test_hash_chain_detects_tampering(tmp_path):
    db = DB(str(tmp_path / "a.db"))
    for i in range(3):
        db.audit("alice", f"act{i}")
    assert db.verify_audit_chain() == {"valid": True, "entries": 3}
    # an attacker with raw DB access drops the triggers and edits a row
    db.conn.execute("DROP TRIGGER audit_no_update")
    db.conn.execute("UPDATE audit_log SET user='mallory' WHERE id=2")
    db.conn.commit()
    res = db.verify_audit_chain()
    assert res["valid"] is False and res["broken_at"] == 2


def test_decide_only_once():
    db = DB(":memory:")
    rid = db.create_result(orthanc_id="x", kind="classification", status="pending_review")
    assert db.decide(rid, "approve", "dr", "ok") is True
    assert db.decide(rid, "reject", "dr", "changed my mind") is False
    assert db.get_result(rid)["status"] == "approved"
