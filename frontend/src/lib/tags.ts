const VALID = /^[a-z0-9-]{1,40}$/;

/** Mirrors backend app/tags.py: trim, lowercase, whitespace -> '-'. Null if invalid. */
export function normalizeTag(raw: string): string | null {
  const name = raw.trim().toLowerCase().replace(/\s+/g, "-");
  return VALID.test(name) ? name : null;
}
