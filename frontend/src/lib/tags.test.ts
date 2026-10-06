import { describe, expect, it } from "vitest";
import { normalizeTag } from "./tags";

describe("normalizeTag", () => {
  it.each([
    ["Goa", "goa"],
    ["  Goa Trip ", "goa-trip"],
    ["new   year", "new-year"],
  ])("normalises %s", (raw, expected) => {
    expect(normalizeTag(raw)).toBe(expected);
  });

  it.each(["", "   ", "#fun!", "café", "x".repeat(41)])("rejects %s", (raw) => {
    expect(normalizeTag(raw)).toBeNull();
  });
});
