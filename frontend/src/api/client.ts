import type {
  ImageDetail, ImagePage, ImagePatch, ListParams, SearchParams, SearchResponse, TagCount,
  UpdateResponse, UploadMeta, UploadResult,
} from "./types";

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

type QueryValue = string | number | undefined | string[];

function qs(params: Record<string, QueryValue>): string {
  const sp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "") continue;
    if (Array.isArray(value)) value.forEach((v) => sp.append(key, v));
    else sp.set(key, String(value));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

function toApiError(status: number, statusText: string, body: unknown): ApiError {
  const err = (body as { error?: { code?: string; message?: string } } | null)?.error;
  return new ApiError(status, err?.code ?? "http_error", err?.message ?? (statusText || "Request failed"));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw toApiError(res.status, res.statusText, body);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  listImages: (p: ListParams) =>
    request<ImagePage>(`/images${qs({ page: p.page, page_size: p.page_size, tags: p.tags, source: p.source, sort: p.sort })}`),

  searchImages: (p: SearchParams) =>
    request<SearchResponse>(`/search${qs({ q: p.q, tags: p.tags, source: p.source, limit: p.limit })}`),

  getImage: (id: string) => request<ImageDetail>(`/images/${id}`),

  updateImage: (id: string, patch: ImagePatch) =>
    request<UpdateResponse>(`/images/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  deleteImage: (id: string) => request<void>(`/images/${id}`, { method: "DELETE" }),

  listTags: () => request<TagCount[]>("/tags"),

  health: () => request<{ status: string }>("/health"),

  /** Upload a single file (the dialog uploads files one by one for per-file progress). */
  uploadImage: (file: File, meta: UploadMeta, onProgress?: (fraction: number) => void) =>
    new Promise<UploadResult>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/images");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress?.(e.loaded / e.total);
      };
      xhr.onload = () => {
        let body: unknown = null;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve((body as UploadResult[])[0]);
        else reject(toApiError(xhr.status, xhr.statusText, body));
      };
      xhr.onerror = () => reject(new ApiError(0, "network_error", "Network error"));
      const form = new FormData();
      form.append("files", file);
      if (meta.tags.length) form.append("tags", meta.tags.join(","));
      if (meta.title.trim()) form.append("title", meta.title.trim());
      if (meta.description.trim()) form.append("description", meta.description.trim());
      xhr.send(form);
    }),
};
