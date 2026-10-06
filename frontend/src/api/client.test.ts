import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./client";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue(
    new Response(status === 204 ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("builds list query with repeated tags and skips empty params", async () => {
    const fetchFn = mockFetch(200, { items: [], page: 1, page_size: 60, total: 0 });
    await api.listImages({ page: 1, page_size: 60, tags: ["goa", "beach"], source: undefined });
    expect(fetchFn).toHaveBeenCalledWith("/api/images?page=1&page_size=60&tags=goa&tags=beach", undefined);
  });

  it("sends PATCH as JSON", async () => {
    const fetchFn = mockFetch(200, { image: {}, warning: null });
    await api.updateImage("abc", { title: "x" });
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("/api/images/abc");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({ title: "x" });
  });

  it("handles 204 on delete", async () => {
    mockFetch(204, null);
    await expect(api.deleteImage("abc")).resolves.toBeUndefined();
  });

  it("raises ApiError from the error envelope", async () => {
    mockFetch(400, { error: { code: "invalid_tag", message: "Invalid tag '#x'" } });
    const err = await api.updateImage("abc", { tags: ["#x"] }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 400, code: "invalid_tag", message: "Invalid tag '#x'" });
  });
});
