import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { ImageSummary } from "../api/types";
import { renderWithClient } from "../test/utils";
import { Gallery } from "./Gallery";

vi.mock("../api/client", () => ({
  api: { listImages: vi.fn(), searchImages: vi.fn() },
  ApiError: class extends Error {},
}));

const img = (id: string, title: string | null = null): ImageSummary => ({
  id, filename: `${id}.png`, title, description: null, tags: [], width: 10, height: 10,
  taken_at: null, created_at: "2026-01-01T00:00:00Z", source: "upload", indexed: true,
  thumb_url: `http://s3/${id}.webp`,
});

beforeEach(() => vi.resetAllMocks());

describe("Gallery", () => {
  it("lists library images and opens one on click", async () => {
    vi.mocked(api.listImages).mockResolvedValue({ items: [img("a", "Beach"), img("b")], page: 1, page_size: 60, total: 2 });
    const onOpen = vi.fn();
    renderWithClient(<Gallery q="" tags={[]} source={undefined} onOpen={onOpen} onUpload={vi.fn()} />);
    await userEvent.click(await screen.findByRole("button", { name: "Beach" }));
    expect(onOpen).toHaveBeenCalledWith("a");
    expect(screen.getByRole("button", { name: "b.png" })).toBeInTheDocument();
  });

  it("shows search results with scores when q is set", async () => {
    vi.mocked(api.searchImages).mockResolvedValue({ items: [{ ...img("a"), score: 0.87 }] });
    renderWithClient(<Gallery q="red car" tags={["goa"]} source="upload" onOpen={vi.fn()} onUpload={vi.fn()} />);
    expect(await screen.findByText("87%")).toBeInTheDocument();
    expect(api.searchImages).toHaveBeenCalledWith({ q: "red car", tags: ["goa"], source: "upload", limit: 60 });
  });

  it("shows the empty library state with an upload action", async () => {
    vi.mocked(api.listImages).mockResolvedValue({ items: [], page: 1, page_size: 60, total: 0 });
    const onUpload = vi.fn();
    renderWithClient(<Gallery q="" tags={[]} source={undefined} onOpen={vi.fn()} onUpload={onUpload} />);
    await userEvent.click(await screen.findByRole("button", { name: "Upload photos" }));
    expect(onUpload).toHaveBeenCalled();
  });

  it("shows no-matches state for an empty search", async () => {
    vi.mocked(api.searchImages).mockResolvedValue({ items: [] });
    renderWithClient(<Gallery q="unicorn" tags={[]} source={undefined} onOpen={vi.fn()} onUpload={vi.fn()} />);
    expect(await screen.findByText("No matching photos")).toBeInTheDocument();
  });
});
