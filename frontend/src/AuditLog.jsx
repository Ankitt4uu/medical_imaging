import { useEffect, useState } from "react";
import { api } from "./api.js";

export default function AuditLog() {
  const [d, setD] = useState(null);
  useEffect(() => { api("/audit?limit=300").then(setD); }, []);
  if (!d) return <p className="muted pad">Loading…</p>;
  return (
    <div className="audit">
      <p className={d.integrity.valid ? "ok" : "error"}>
        {d.integrity.valid ? `Chain verified: ${d.integrity.entries} entries, no tampering detected.` : `Chain BROKEN at entry #${d.integrity.broken_at}.`}
      </p>
      <table>
        <thead><tr><th>#</th><th>Time (UTC)</th><th>User</th><th>Action</th><th>Result</th><th>Model</th><th>Detail</th></tr></thead>
        <tbody>
          {d.entries.map((e) => (
            <tr key={e.id}><td>{e.id}</td><td>{e.ts}</td><td>{e.user}</td><td>{e.action}</td><td>{e.result_id ?? ""}</td><td>{e.model_version ?? ""}</td><td>{e.detail ?? ""}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
