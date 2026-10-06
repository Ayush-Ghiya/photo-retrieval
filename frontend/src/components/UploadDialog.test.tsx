import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { renderWithClient } from "../test/utils";
import { UploadDialog } from "./UploadDialog";

vi.mock("../api/client", () => ({
  api: { uploadImage: vi.fn(), listTags: vi.fn() },
  ApiError: class extends Error {},
}));

const file = (name: string) => new File([new Uint8Array([1, 2, 3])], name, { type: "image/png" });

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.listTags).mockResolvedValue([]);
});

describe("UploadDialog", () => {
  it("renders nothing when closed", () => {
    renderWithClient(<UploadDialog open={false} onClose={vi.fn()} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("uploads each file with shared metadata and shows per-file results", async () => {
    vi.mocked(api.uploadImage)
      .mockResolvedValueOnce({ filename: "a.png", status: "created", id: "1", message: null })
      .mockResolvedValueOnce({ filename: "b.png", status: "duplicate", id: "2", message: "Already in your library" })
      .mockRejectedValueOnce(new Error("Network error"));
    renderWithClient(<UploadDialog open onClose={vi.fn()} />);

    await userEvent.upload(screen.getByLabelText("Choose files"), [file("a.png"), file("b.png"), file("c.png")]);
    await userEvent.type(screen.getByLabelText("Tags for upload"), "Goa Trip{Enter}");
    await userEvent.type(screen.getByLabelText("Title"), "Holiday");
    await userEvent.click(screen.getByRole("button", { name: "Upload 3 files" }));

    expect(await screen.findByText("Added")).toBeInTheDocument();
    expect(await screen.findByText("Duplicate")).toBeInTheDocument();
    expect(await screen.findByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("1 added · 1 duplicate · 1 failed")).toBeInTheDocument();
    expect(api.uploadImage).toHaveBeenCalledTimes(3);
    expect(vi.mocked(api.uploadImage).mock.calls[0][1]).toEqual({ tags: ["goa-trip"], title: "Holiday", description: "" });
  });

  it("disables upload with no files", () => {
    renderWithClient(<UploadDialog open onClose={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Upload 0 files" })).toBeDisabled();
  });
});
