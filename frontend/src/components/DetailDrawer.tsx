import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { api } from "../api/client";
import type { ImageDetail, ImagePatch } from "../api/types";
import { TagInput } from "./TagInput";

interface DetailDrawerProps {
  imageId: string | null;
  onClose: () => void;
}

export function DetailDrawer({ imageId, onClose }: DetailDrawerProps) {
  const query = useQuery({
    queryKey: ["image", imageId],
    queryFn: () => api.getImage(imageId!),
    enabled: imageId !== null,
  });

  useEffect(() => {
    if (!imageId) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [imageId, onClose]);

  if (!imageId) return null;
  return (
    <div className="fixed inset-0 z-20 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <aside role="dialog" aria-label="Photo details" className="relative flex h-full w-full max-w-xl flex-col overflow-y-auto bg-white shadow-xl">
        <button type="button" aria-label="Close" onClick={onClose} className="absolute right-3 top-3 z-10 rounded-full bg-white/80 px-2 text-xl">
          ×
        </button>
        {query.isPending && <p className="p-6 text-stone-500">Loading…</p>}
        {query.isError && <p className="p-6 text-red-600">{query.error.message}</p>}
        {query.data && <DetailForm key={query.data.id} image={query.data} onClose={onClose} />}
      </aside>
    </div>
  );
}

function sameTags(a: string[], b: string[]) {
  return a.length === b.length && a.every((t, i) => t === b[i]);
}

function DetailForm({ image, onClose }: { image: ImageDetail; onClose: () => void }) {
  const qc = useQueryClient();
  const [title, setTitle] = useState(image.title ?? "");
  const [description, setDescription] = useState(image.description ?? "");
  const [tags, setTags] = useState(image.tags);
  const tagList = useQuery({ queryKey: ["tags"], queryFn: api.listTags });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["images"] });
    qc.invalidateQueries({ queryKey: ["search"] });
    qc.invalidateQueries({ queryKey: ["tags"] });
  };

  const save = useMutation({
    mutationFn: (patch: ImagePatch) => api.updateImage(image.id, patch),
    onSuccess: (res) => {
      qc.setQueryData(["image", image.id], res.image);
      invalidate();
      if (res.warning) toast.warning(res.warning);
      else toast.success("Saved");
    },
    onError: (e) => toast.error(e.message),
  });

  const remove = useMutation({
    mutationFn: () => api.deleteImage(image.id),
    onSuccess: () => {
      qc.removeQueries({ queryKey: ["image", image.id] });
      invalidate();
      toast.success("Photo deleted");
      onClose();
    },
    onError: (e) => toast.error(e.message),
  });

  function onSave() {
    const patch: ImagePatch = {};
    if (title.trim() !== (image.title ?? "")) patch.title = title.trim() || null;
    if (description.trim() !== (image.description ?? "")) patch.description = description.trim() || null;
    if (!sameTags(tags, image.tags)) patch.tags = tags;
    if (Object.keys(patch).length === 0) {
      toast("Nothing to save");
      return;
    }
    save.mutate(patch);
  }

  function onDelete() {
    if (window.confirm("Delete this photo permanently?")) remove.mutate();
  }

  const field = "w-full rounded-lg border border-stone-300 px-3 py-2 outline-none focus:ring-2 focus:ring-amber-400";
  return (
    <div className="flex flex-col">
      <a href={image.original_url} target="_blank" rel="noreferrer" className="bg-stone-900">
        <img src={image.original_url} alt={image.title || image.filename} className="max-h-[60vh] w-full object-contain" />
      </a>
      <div className="space-y-4 p-6">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm text-stone-600">
          <dt>File</dt>
          <dd className="truncate">{image.filename}</dd>
          <dt>Size</dt>
          <dd>
            {image.width} × {image.height}
          </dd>
          <dt>Taken</dt>
          <dd>{image.taken_at ? new Date(image.taken_at).toLocaleString() : "—"}</dd>
          {!image.indexed && (
            <>
              <dt>Search</dt>
              <dd className="text-amber-700">Not indexed yet</dd>
            </>
          )}
        </dl>
        <label className="block text-sm font-medium">
          Title
          <input className={`${field} mt-1`} value={title} maxLength={200} onChange={(e) => setTitle(e.target.value)} />
        </label>
        <label className="block text-sm font-medium">
          Description
          <textarea className={`${field} mt-1`} rows={3} value={description} maxLength={2000} onChange={(e) => setDescription(e.target.value)} />
        </label>
        <div className="text-sm font-medium">
          <span className="mb-1 block">Tags</span>
          <TagInput label="Tags" value={tags} onChange={setTags} suggestions={tagList.data?.map((t) => t.name) ?? []} placeholder="Add a tag and press Enter" />
        </div>
        <div className="flex justify-between pt-2">
          <button type="button" onClick={onDelete} disabled={remove.isPending} className="rounded-lg px-4 py-2 text-red-600 hover:bg-red-50">
            Delete
          </button>
          <button type="button" onClick={onSave} disabled={save.isPending} className="rounded-lg bg-amber-500 px-4 py-2 font-medium text-white hover:bg-amber-600 disabled:opacity-50">
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
