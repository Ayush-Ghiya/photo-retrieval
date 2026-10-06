import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { api } from "../api/client";
import type { Source } from "../api/types";
import { ImageCard } from "./ImageCard";

const PAGE_SIZE = 60;

interface GalleryProps {
  q: string;
  tags: string[];
  source: Source | undefined;
  onOpen: (id: string) => void;
  onUpload: () => void;
}

const grid = "grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6";

export function Gallery({ q, tags, source, onOpen, onUpload }: GalleryProps) {
  const searching = q.trim().length > 0;

  const search = useQuery({
    queryKey: ["search", q, tags, source],
    queryFn: () => api.searchImages({ q, tags, source, limit: PAGE_SIZE }),
    enabled: searching,
  });

  const library = useInfiniteQuery({
    queryKey: ["images", "list", tags, source],
    queryFn: ({ pageParam }) => api.listImages({ page: pageParam, page_size: PAGE_SIZE, tags, source }),
    initialPageParam: 1,
    getNextPageParam: (last) => (last.page * last.page_size < last.total ? last.page + 1 : undefined),
    enabled: !searching,
  });

  const sentinel = useRef<HTMLDivElement>(null);
  const { hasNextPage, fetchNextPage, isFetchingNextPage } = library;
  useEffect(() => {
    const el = sentinel.current;
    if (!el || !("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) fetchNextPage();
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasNextPage, fetchNextPage, isFetchingNextPage]);

  const active = searching ? search : library;
  if (active.isPending) return <p className="py-16 text-center text-stone-500">Loading…</p>;
  if (active.isError) return <p className="py-16 text-center text-red-600">{active.error.message}</p>;

  if (searching) {
    const items = search.data?.items ?? [];
    if (!items.length) return <p className="py-16 text-center text-stone-500">No matching photos</p>;
    return (
      <div className={grid}>
        {items.map((item) => (
          <ImageCard key={item.id} image={item} score={item.score} onOpen={onOpen} />
        ))}
      </div>
    );
  }

  const items = library.data?.pages.flatMap((p) => p.items) ?? [];
  if (!items.length) {
    if (tags.length) return <p className="py-16 text-center text-stone-500">No matching photos</p>;
    return (
      <div className="py-16 text-center">
        <p className="mb-4 text-stone-500">No photos yet.</p>
        <button type="button" onClick={onUpload} className="rounded-lg bg-amber-500 px-4 py-2 font-medium text-white hover:bg-amber-600">
          Upload photos
        </button>
      </div>
    );
  }
  return (
    <>
      <div className={grid}>
        {items.map((item) => (
          <ImageCard key={item.id} image={item} onOpen={onOpen} />
        ))}
      </div>
      <div ref={sentinel} className="h-8" />
      {isFetchingNextPage && <p className="py-4 text-center text-stone-500">Loading more…</p>}
    </>
  );
}
