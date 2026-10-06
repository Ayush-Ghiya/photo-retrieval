import { useEffect, useState } from "react";
import type { SearchState } from "../hooks/useSearchState";
import { TagInput } from "./TagInput";

interface SearchBarProps {
  q: string;
  tags: string[];
  suggestions: string[];
  onChange: (next: Partial<SearchState>) => void;
}

export function SearchBar({ q, tags, suggestions, onChange }: SearchBarProps) {
  const [text, setText] = useState(q);

  useEffect(() => setText(q), [q]);

  useEffect(() => {
    if (text.trim() === q) return;
    const timer = setTimeout(() => onChange({ q: text.trim() }), 400);
    return () => clearTimeout(timer);
  }, [text, q, onChange]);

  return (
    <form
      role="search"
      className="flex flex-1 flex-wrap items-start gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        onChange({ q: text.trim() });
      }}
    >
      <input
        aria-label="Search photos"
        type="search"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder='Describe a photo, e.g. "red car at night"'
        className="min-w-64 flex-1 rounded-lg border border-stone-300 bg-white px-4 py-2 outline-none focus:ring-2 focus:ring-amber-400"
      />
      <div className="w-72">
        <TagInput
          label="Filter by tag"
          placeholder="Filter by tag"
          value={tags}
          suggestions={suggestions}
          onChange={(next) => onChange({ tags: next })}
        />
      </div>
    </form>
  );
}
