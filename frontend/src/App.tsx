import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Toaster } from "sonner";
import { api } from "./api/client";
import type { Source } from "./api/types";
import { DetailDrawer } from "./components/DetailDrawer";
import { Gallery } from "./components/Gallery";
import { HealthBanner } from "./components/HealthBanner";
import { SearchBar } from "./components/SearchBar";
import { UploadDialog } from "./components/UploadDialog";
import { useSearchState } from "./hooks/useSearchState";

export default function App() {
  const [search, setSearch] = useSearchState();
  const [openId, setOpenId] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [demoChoice, setDemoChoice] = useState<boolean | null>(null);

  const tags = useQuery({ queryKey: ["tags"], queryFn: api.listTags });
  const ownCount = useQuery({
    queryKey: ["images", "own-count"],
    queryFn: () => api.listImages({ page: 1, page_size: 1, source: "upload" }),
  });
  // Show demo images by default until the user has uploaded their own.
  const showDemo = demoChoice ?? (ownCount.data ? ownCount.data.total === 0 : true);
  const source: Source | undefined = showDemo ? undefined : "upload";

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900">
      <HealthBanner />
      <header className="sticky top-0 z-10 border-b border-stone-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-start gap-3 px-4 py-3">
          <h1 className="py-2 text-lg font-semibold">Photos</h1>
          <SearchBar q={search.q} tags={search.tags} suggestions={tags.data?.map((t) => t.name) ?? []} onChange={setSearch} />
          <label className="flex items-center gap-2 py-2 text-sm text-stone-600">
            <input type="checkbox" checked={showDemo} onChange={(e) => setDemoChoice(e.target.checked)} />
            Show demo images
          </label>
          <button
            type="button"
            onClick={() => setUploadOpen(true)}
            className="rounded-lg bg-amber-500 px-4 py-2 font-medium text-white hover:bg-amber-600"
          >
            Upload
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Gallery q={search.q} tags={search.tags} source={source} onOpen={setOpenId} onUpload={() => setUploadOpen(true)} />
      </main>
      <DetailDrawer imageId={openId} onClose={() => setOpenId(null)} />
      <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} />
      <Toaster position="bottom-right" richColors />
    </div>
  );
}
