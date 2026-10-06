import { useId, useState, type KeyboardEvent } from "react";
import { normalizeTag } from "../lib/tags";

interface TagInputProps {
  label: string;
  value: string[];
  onChange: (tags: string[]) => void;
  suggestions?: string[];
  placeholder?: string;
}

export function TagInput({ label, value, onChange, suggestions = [], placeholder }: TagInputProps) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const listId = useId();

  const prefix = text.trim().toLowerCase();
  const matches = prefix
    ? suggestions.filter((s) => s.startsWith(prefix) && !value.includes(s)).slice(0, 8)
    : [];

  function add(raw: string) {
    const tag = normalizeTag(raw);
    if (!tag) {
      setError("Tags use letters, digits and hyphens (max 40 characters)");
      return;
    }
    setError(null);
    if (!value.includes(tag)) onChange([...value, tag]);
    setText("");
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if ((e.key === "Enter" || e.key === ",") && text.trim()) {
      e.preventDefault();
      add(text);
    } else if (e.key === "Backspace" && !text && value.length) {
      onChange(value.slice(0, -1));
    }
  }

  return (
    <div className="relative">
      <div className="flex flex-wrap items-center gap-1 rounded-lg border border-stone-300 bg-white px-2 py-1 focus-within:ring-2 focus-within:ring-amber-400">
        {value.map((tag) => (
          <span key={tag} className="flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-sm text-amber-900">
            {tag}
            <button
              type="button"
              aria-label={`Remove tag ${tag}`}
              className="text-amber-700 hover:text-amber-950"
              onClick={() => onChange(value.filter((t) => t !== tag))}
            >
              ×
            </button>
          </span>
        ))}
        <input
          aria-label={label}
          aria-controls={listId}
          className="min-w-24 flex-1 bg-transparent py-1 text-sm outline-none"
          value={text}
          placeholder={value.length ? "" : placeholder}
          onChange={(e) => {
            setText(e.target.value.replace(",", ""));
            setError(null);
          }}
          onKeyDown={onKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => {
            setFocused(false);
            // Typed-but-unconfirmed text counts: commit it so Save/Upload never drops it silently.
            if (text.trim()) add(text);
          }}
        />
      </div>
      {error && (
        <p role="alert" className="mt-1 text-xs text-red-600">
          {error}
        </p>
      )}
      {focused && matches.length > 0 && (
        <ul id={listId} role="listbox" className="absolute z-30 mt-1 w-full rounded-lg border bg-white py-1 shadow-lg">
          {matches.map((s) => (
            <li
              key={s}
              role="option"
              aria-selected={false}
              className="cursor-pointer px-3 py-1 text-sm hover:bg-amber-50"
              onMouseDown={(e) => {
                e.preventDefault();
                add(s);
              }}
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
