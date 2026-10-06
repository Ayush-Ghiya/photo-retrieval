import type { ImageSummary } from "../api/types";

interface ImageCardProps {
  image: ImageSummary;
  score?: number | null;
  onOpen: (id: string) => void;
}

export function ImageCard({ image, score, onOpen }: ImageCardProps) {
  const name = image.title || image.filename;
  return (
    <button
      type="button"
      aria-label={name}
      onClick={() => onOpen(image.id)}
      className="group relative aspect-square overflow-hidden rounded-lg bg-stone-200 focus:outline-none focus:ring-2 focus:ring-amber-400"
    >
      <img src={image.thumb_url} alt={name} loading="lazy" className="h-full w-full object-cover transition group-hover:scale-105" />
      {score != null && (
        <span className="absolute right-1 top-1 rounded bg-black/60 px-1.5 py-0.5 text-xs text-white opacity-0 group-hover:opacity-100">
          {Math.round(score * 100)}%
        </span>
      )}
      {image.tags.length > 0 && (
        <span className="absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-black/60 px-2 pb-1 pt-4 text-left text-xs text-white opacity-0 group-hover:opacity-100">
          {image.tags.join(" · ")}
        </span>
      )}
    </button>
  );
}
