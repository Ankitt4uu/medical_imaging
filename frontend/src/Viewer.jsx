import { useEffect, useState } from "react";
import { api, useAuthImage } from "./api.js";

export default function Viewer({ id, onChanged }) {
  const [r, setR] = useState(null);
  const [opacity, setOpacity] = useState(0.7);
  const [show, setShow] = useState(true);
  const [slice, setSlice] = useState(0);
  const [comment, setComment] = useState("");
  const [err, setErr] = useState("");

  useEffect(() => { api(`/results/${id}`).then((x) => { setR(x); if (x.kind === "segmentation") setSlice(Math.floor((x.artifacts?.instances?.length || 2) / 2)); }); }, [id]);
  const base = useAuthImage(r ? `/results/${id}/image?slice=${slice}` : null);
  const over = useAuthImage(r ? `/results/${id}/overlay?slice=${slice}` : null);
  if (!r) return <p className="muted pad">Loading…</p>;

  const seg = r.kind === "segmentation";
  const p = r.prediction || {};
  const pending = r.status === "pending_review";

  async function decide(decision) {
    setErr("");
    try { setR(await api(`/results/${id}/decision`, { method: "POST", body: { decision, comment } })); onChanged(); }
    catch (x) { setErr(x.message); }
  }

  return (
    <div className="viewer">
      <div className="stage">
        {r.demo_mode ? <div className="banner warn" role="alert">Demo model: weights are untrained, so predictions and heatmaps are meaningless.</div> : null}
        <div className="canvas">
          {base ? <img src={base} alt="Scan" /> : <p className="muted pad">Scan unavailable (is Orthanc running?)</p>}
          {show && over && <img className="overlay" src={over} alt="AI overlay" style={{ opacity }} />}
        </div>
        <div className="controls">
          <label><input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} /> {seg ? "Segmentation" : "Explanation heatmap"}</label>
          <label>Opacity <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} /></label>
          {seg && <label>Slice {slice + 1}/{r.artifacts.instances.length}
            <input type="range" min="0" max={r.artifacts.instances.length - 1} value={slice} onChange={(e) => setSlice(+e.target.value)} /></label>}
        </div>
      </div>
      <div className="panel">
        <h2>{seg ? `${p.label} segmentation` : "AI findings"}</h2>
        <p className="muted">Model {r.model_version}</p>
        {seg ? (
          <dl><dt>Volume</dt><dd>{p.volume_ml} ml</dd><dt>Slices with mask</dt><dd>{p.slices_with_mask} of {p.n_slices}</dd></dl>
        ) : p.probabilities ? (
          <>
            <p className="top"><strong>{p.top_class}</strong> <span>{(p.confidence * 100).toFixed(1)}% confidence</span></p>
            <p className="muted">Heatmap shows where the model looked for {p.gradcam_target}.</p>
            <ul className="bars">
              {Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([c, v]) => (
                <li key={c} className={p.flagged?.includes(c) ? "flag" : ""}>
                  <span>{c.replace("_", " ")}</span>
                  <div className="bar"><i style={{ width: `${v * 100}%` }} /></div>
                  <span>{(v * 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>
          </>
        ) : <p className="error">{r.error || "No prediction"}</p>}

        <div className="decision">
          <h3>Radiologist decision</h3>
          {pending ? (
            <>
              <textarea placeholder="Comment (optional)" value={comment} onChange={(e) => setComment(e.target.value)} />
              <div className="btns">
                <button className="approve" onClick={() => decide("approve")}>Approve AI result</button>
                <button className="reject" onClick={() => decide("reject")}>Reject AI result</button>
              </div>
            </>
          ) : (
            <p>{r.status === "approved" ? "Approved" : "Rejected"} by <strong>{r.decided_by}</strong> at {r.decided_at}.{r.decision_comment && <><br />“{r.decision_comment}”</>}</p>
          )}
          {err && <p className="error" role="alert">{err}</p>}
        </div>
      </div>
    </div>
  );
}
