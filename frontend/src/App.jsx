import { useCallback, useEffect, useState } from "react";
import { api, getToken, setToken } from "./api.js";
import Viewer from "./Viewer.jsx";
import AuditLog from "./AuditLog.jsx";

const NOTICE = "For education only. Not for clinical use.";

function Login({ onDone }) {
  const [u, setU] = useState("radiologist");
  const [p, setP] = useState("");
  const [err, setErr] = useState("");
  async function submit(e) {
    e.preventDefault();
    try {
      const r = await api("/login", { method: "POST", body: { username: u, password: p } });
      setToken(r.token);
      onDone(r.user);
    } catch (x) { setErr(x.message); }
  }
  return (
    <main className="login">
      <form onSubmit={submit}>
        <h1>Imaging review workstation</h1>
        <p className="muted">{NOTICE}</p>
        <label>Username<input value={u} onChange={(e) => setU(e.target.value)} autoFocus /></label>
        <label>Password<input type="password" value={p} onChange={(e) => setP(e.target.value)} /></label>
        {err && <p className="error" role="alert">{err}</p>}
        <button className="primary">Sign in</button>
      </form>
    </main>
  );
}

const STATUS = { pending_review: "Needs review", approved: "Approved", rejected: "Rejected", failed: "Failed" };

export default function App() {
  const [user, setUser] = useState(getToken() ? "radiologist" : null);
  const [results, setResults] = useState([]);
  const [sel, setSel] = useState(null);
  const [tab, setTab] = useState("queue");
  const [filter, setFilter] = useState("");
  const [msg, setMsg] = useState("");

  const load = useCallback(() => api(`/results${filter ? `?status=${filter}` : ""}`).then(setResults).catch(() => {}), [filter]);
  useEffect(() => { if (user) { load(); const t = setInterval(load, 5000); return () => clearInterval(t); } }, [user, load]);

  async function upload(e) {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try { await api("/upload", { method: "POST", body: fd }); setMsg(`Uploaded ${f.name}. Analysis starts in a few seconds.`); }
    catch (x) { setMsg(x.message); }
    e.target.value = "";
  }

  if (!user) return <Login onDone={setUser} />;
  const summary = (r) => r.kind === "segmentation" ? `${r.prediction?.label ?? ""} ${r.prediction?.volume_ml ?? ""} ml` : r.prediction ? `${r.prediction.top_class} ${(r.prediction.confidence * 100).toFixed(0)}%` : r.error;

  return (
    <div className="shell">
      <header>
        <strong>Imaging review</strong>
        <nav>
          <button className={tab === "queue" ? "on" : ""} onClick={() => setTab("queue")}>Review queue</button>
          <button className={tab === "audit" ? "on" : ""} onClick={() => setTab("audit")}>Audit log</button>
        </nav>
        <label className="upload">Upload scan<input type="file" accept=".dcm,.png,.jpg,.jpeg,application/dicom" onChange={upload} hidden /></label>
        <button onClick={() => { setToken(""); setUser(null); }}>Sign out</button>
      </header>
      <div className="banner" role="note">{NOTICE} Research prototype; outputs must not inform patient care.</div>
      {msg && <div className="toast" onClick={() => setMsg("")}>{msg}</div>}
      {tab === "audit" ? <AuditLog /> : (
        <div className="split">
          <aside>
            <select value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Filter by status">
              <option value="">All studies</option>
              <option value="pending_review">Needs review</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
            </select>
            {results.length === 0 && <p className="muted pad">No studies yet. Upload a scan, or push studies to Orthanc.</p>}
            <ul>
              {results.map((r) => (
                <li key={r.id}>
                  <button className={sel === r.id ? "row on" : "row"} onClick={() => setSel(r.id)}>
                    <span>#{r.id} · {r.patient_id || "unknown"} · {r.modality}</span>
                    <span className={`pill ${r.status}`}>{STATUS[r.status]}</span>
                    <span className="muted">{summary(r)}{r.demo_mode ? " (demo model)" : ""}</span>
                  </button>
                </li>
              ))}
            </ul>
          </aside>
          <section>{sel ? <Viewer key={sel} id={sel} onChanged={load} /> : <p className="muted pad">Select a study to review.</p>}</section>
        </div>
      )}
    </div>
  );
}
