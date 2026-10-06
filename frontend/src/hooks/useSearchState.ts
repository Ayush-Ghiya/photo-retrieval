import { useCallback, useEffect, useState } from "react";

export interface SearchState {
  q: string;
  tags: string[];
}

function read(search: string): SearchState {
  const sp = new URLSearchParams(search);
  return { q: sp.get("q") ?? "", tags: sp.getAll("tags") };
}

function toUrl(state: SearchState): string {
  const sp = new URLSearchParams();
  if (state.q) sp.set("q", state.q);
  state.tags.forEach((t) => sp.append("tags", t));
  const s = sp.toString();
  return s ? `?${s}` : window.location.pathname;
}

export function useSearchState() {
  const [state, setState] = useState<SearchState>(() => read(window.location.search));

  useEffect(() => {
    const onPop = () => setState(read(window.location.search));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    window.history.replaceState(null, "", toUrl(state));
  }, [state]);

  const update = useCallback((next: Partial<SearchState>) => setState((prev) => ({ ...prev, ...next })), []);
  return [state, update] as const;
}
