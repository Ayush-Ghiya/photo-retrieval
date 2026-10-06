import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { TagInput } from "./TagInput";

type Status = "pending" | "uploading" | "created" | "duplicate" | "error";

interface Entry {
  file: File;
  preview: string;
  status: Status;
  progress: number;
  message: string | null;
}

const STATUS_LABEL: Record<Status, string> = {
  pending: "Ready",
  uploading: "Uploading…",
  created: "Added",
  duplicate: "Duplicate",
  error: "Failed",
};

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
}

export function UploadDialog({ open, onClose }: UploadDialogProps) {
  const qc = useQueryClient();
  const [entries, setEntries] = useState<Entry[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const tagList = useQuery({ queryKey: ["tags"], queryFn: api.listTags, enabled: open });
  const previews = useRef<string[]>([]);

  // Revoke preview object URLs only on unmount (close() revokes them too).
  useEffect(() => () => previews.current.forEach((url) => URL.revokeObjectURL(url)), []);

  if (!open) return null;

  function addFiles(files: FileList | File[]) {
    const next = Array.from(files).map((file) => ({
      file,
      preview: typeof URL.createObjectURL === "function" ? URL.createObjectURL(file) : "",
      status: "pending" as Status,
      progress: 0,
      message: null,
    }));
    previews.current.push(...next.map((e) => e.preview).filter(Boolean));
    setEntries((prev) => [...prev, ...next]);
  }

  function patch(index: number, changes: Partial<Entry>) {
    setEntries((prev) => prev.map((e, i) => (i === index ? { ...e, ...changes } : e)));
  }

  async function uploadAll() {
    setBusy(true);
    const meta = { tags, title, description };
    for (let i = 0; i < entries.length; i++) {
      if (entries[i].status !== "pending") continue;
      patch(i, { status: "uploading" });
      try {
        const res = await api.uploadImage(entries[i].file, meta, (p) => patch(i, { progress: p }));
        patch(i, { status: res.status, progress: 1, message: res.message });
      } catch (e) {
        patch(i, { status: "error", message: (e as Error).message });
      }
    }
    setBusy(false);
    qc.invalidateQueries({ queryKey: ["images"] });
    qc.invalidateQueries({ queryKey: ["search"] });
    qc.invalidateQueries({ queryKey: ["tags"] });
  }

  function close() {
    if (busy) return;
    previews.current.forEach((url) => URL.revokeObjectURL(url));
    previews.current = [];
    setEntries([]);
    setTags([]);
    setTitle("");
    setDescription("");
    onClose();
  }

  const pending = entries.filter((e) => e.status === "pending").length;
  const done = entries.filter((e) => ["created", "duplicate", "error"].includes(e.status));
  const count = (s: Status) => entries.filter((e) => e.status === s).length;
  const field = "mt-1 w-full rounded-lg border border-stone-300 px-3 py-2 outline-none focus:ring-2 focus:ring-amber-400";

  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={close} />
      <div role="dialog" aria-label="Upload photos" className="relative max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white p-6 shadow-xl">
        <h2 className="mb-4 text-lg font-semibold">Upload photos</h2>

        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
          className={`flex cursor-pointer flex-col items-center rounded-lg border-2 border-dashed p-8 text-stone-500 ${dragging ? "border-amber-400 bg-amber-50" : "border-stone-300"}`}
        >
          Drag photos here or click to choose
          <input
            type="file"
            multiple
            accept="image/jpeg,image/png,image/webp,image/heic,.heic"
            aria-label="Choose files"
            className="sr-only"
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />
        </label>

        {entries.length > 0 && (
          <ul className="mt-4 max-h-60 space-y-2 overflow-y-auto">
            {entries.map((e, i) => (
              <li key={i} className="flex items-center gap-3 text-sm">
                {e.preview ? <img src={e.preview} alt="" className="h-10 w-10 rounded object-cover" /> : <div className="h-10 w-10 rounded bg-stone-200" />}
                <div className="min-w-0 flex-1">
                  <p className="truncate">{e.file.name}</p>
                  {e.status === "uploading" && (
                    <div className="h-1 rounded bg-stone-200">
                      <div className="h-1 rounded bg-amber-500" style={{ width: `${Math.round(e.progress * 100)}%` }} />
                    </div>
                  )}
                  {e.message && <p className="truncate text-xs text-stone-500">{e.message}</p>}
                </div>
                <span className={e.status === "error" ? "text-red-600" : e.status === "created" ? "text-green-700" : "text-stone-600"}>
                  {STATUS_LABEL[e.status]}
                </span>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-4 space-y-3">
          <div className="text-sm font-medium">
            <span className="mb-1 block">Tags (applied to all)</span>
            <TagInput label="Tags for upload" value={tags} onChange={setTags} suggestions={tagList.data?.map((t) => t.name) ?? []} placeholder="Add a tag and press Enter" />
          </div>
          <label className="block text-sm font-medium">
            Title
            <input className={field} value={title} maxLength={200} onChange={(e) => setTitle(e.target.value)} />
          </label>
          <label className="block text-sm font-medium">
            Description
            <textarea className={field} rows={2} value={description} maxLength={2000} onChange={(e) => setDescription(e.target.value)} />
          </label>
        </div>

        <div className="mt-6 flex items-center justify-between">
          <p className="text-sm text-stone-600">
            {done.length > 0 && `${count("created")} added · ${count("duplicate")} duplicate · ${count("error")} failed`}
          </p>
          <div className="flex gap-2">
            <button type="button" onClick={close} disabled={busy} className="rounded-lg px-4 py-2 hover:bg-stone-100">
              {done.length ? "Done" : "Cancel"}
            </button>
            <button
              type="button"
              onClick={uploadAll}
              disabled={busy || pending === 0}
              className="rounded-lg bg-amber-500 px-4 py-2 font-medium text-white hover:bg-amber-600 disabled:opacity-50"
            >
              {`Upload ${pending} file${pending === 1 ? "" : "s"}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
