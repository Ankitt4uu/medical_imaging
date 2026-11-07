import { useEffect, useState } from "react";

let token = sessionStorage.getItem("token") || "";
export const getToken = () => token;
export const setToken = (t) => {
  token = t;
  t ? sessionStorage.setItem("token", t) : sessionStorage.removeItem("token");
};

export async function api(path, opts = {}) {
  const res = await fetch(`/api${path}`, {
    ...opts,
    headers: {
      ...(opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  if (res.status === 401 && token) { setToken(""); window.location.reload(); }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

/** <img> tags can't send an Authorization header, so fetch as blob and hand back an object URL. */
export function useAuthImage(path) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!path) return;
    let alive = true, obj = null;
    fetch(`/api${path}`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => (r.ok ? r.blob() : null))
      .then((b) => { if (alive && b) { obj = URL.createObjectURL(b); setUrl(obj); } });
    return () => { alive = false; obj && URL.revokeObjectURL(obj); };
  }, [path]);
  return url;
}
