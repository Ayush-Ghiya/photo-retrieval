import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { ImageDetail } from "../api/types";
import { renderWithClient } from "../test/utils";
import { DetailDrawer } from "./DetailDrawer";

vi.mock("../api/client", () => ({
  api: { getImage: vi.fn(), updateImage: vi.fn(), deleteImage: vi.fn(), listTags: vi.fn() },
  ApiError: class extends Error {},
}));

const detail: ImageDetail = {
  id: "a", filename: "a.jpg", title: "Old", description: "desc", tags: ["goa"], width: 4000, height: 3000,
  taken_at: "2024-05-01T10:20:30Z", created_at: "2026-01-01T00:00:00Z", source: "upload", indexed: true,
  thumb_url: "http://s3/t.webp", original_url: "http://s3/o.jpg", mime_type: "image/jpeg",
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.getImage).mockResolvedValue(detail);
  vi.mocked(api.listTags).mockResolvedValue([{ name: "goa", count: 1 }]);
});

describe("DetailDrawer", () => {
  it("renders nothing without an image id", () => {
    renderWithClient(<DetailDrawer imageId={null} onClose={vi.fn()} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("shows details and saves only changed fields", async () => {
    vi.mocked(api.updateImage).mockResolvedValue({ image: { ...detail, title: "New" }, warning: null });
    renderWithClient(<DetailDrawer imageId="a" onClose={vi.fn()} />);
    const title = await screen.findByLabelText("Title");
    expect(title).toHaveValue("Old");
    expect(screen.getByText("4000 × 3000")).toBeInTheDocument();
    await userEvent.clear(title);
    await userEvent.type(title, "New");
    await userEvent.type(screen.getByLabelText("Tags"), "Beach Day{Enter}");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(api.updateImage).toHaveBeenCalledWith("a", { title: "New", tags: ["goa", "beach-day"] });
  });

  it("deletes after confirmation and closes", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(api.deleteImage).mockResolvedValue(undefined);
    const onClose = vi.fn();
    renderWithClient(<DetailDrawer imageId="a" onClose={onClose} />);
    await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(api.deleteImage).toHaveBeenCalledWith("a");
    await vi.waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it("does not delete when confirmation is cancelled", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    renderWithClient(<DetailDrawer imageId="a" onClose={vi.fn()} />);
    await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(api.deleteImage).not.toHaveBeenCalled();
  });
});
