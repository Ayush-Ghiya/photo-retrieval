import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";

// Requires: docker compose up -d, and the API running on :8080 (see README).
test("upload → tag → search finds it", async ({ page, request }) => {
  const tag = `smoke-${Date.now()}`;
  // Trailing bytes after IEND keep the PNG valid but make the hash unique per run.
  const png = Buffer.concat([readFileSync(path.join(import.meta.dirname, "fixtures/red.png")), Buffer.from(tag)]);

  await page.goto("/");
  await page.getByRole("button", { name: "Upload", exact: true }).click();
  await page.getByLabel("Choose files").setInputFiles({ name: "red.png", mimeType: "image/png", buffer: png });
  const tagBox = page.getByLabel("Tags for upload");
  await tagBox.fill(tag);
  await tagBox.press("Enter");
  await page.getByRole("button", { name: "Upload 1 file" }).click();
  await expect(page.getByText("Added", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Done" }).click();

  const filter = page.getByLabel("Filter by tag");
  await filter.fill(tag);
  await filter.press("Enter");
  await page.getByLabel("Search photos").fill("a red square");
  await page.getByLabel("Search photos").press("Enter");
  await expect(page.getByRole("button", { name: "red.png" })).toBeVisible();

  // Cleanup
  const list = await (await request.get(`/api/images?tags=${tag}`)).json();
  for (const item of list.items) await request.delete(`/api/images/${item.id}`);
});
