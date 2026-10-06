import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { TagInput } from "./TagInput";

function Harness({ initial = [] as string[], suggestions = [] as string[] }) {
  const [tags, setTags] = useState(initial);
  return (
    <>
      <TagInput label="Tags" value={tags} onChange={setTags} suggestions={suggestions} />
      <output data-testid="value">{tags.join("|")}</output>
    </>
  );
}

describe("TagInput", () => {
  it("adds normalised tags on Enter and comma, ignoring duplicates", async () => {
    render(<Harness />);
    const input = screen.getByLabelText("Tags");
    await userEvent.type(input, "Goa Trip{Enter}family,goa trip{Enter}");
    expect(screen.getByTestId("value")).toHaveTextContent("goa-trip|family");
    expect(input).toHaveValue("");
  });

  it("shows an error for invalid tags", async () => {
    render(<Harness />);
    await userEvent.type(screen.getByLabelText("Tags"), "#fun!{Enter}");
    expect(screen.getByRole("alert")).toHaveTextContent(/letters, digits/);
    expect(screen.getByTestId("value")).toHaveTextContent("");
  });

  it("removes with the chip button and with Backspace on empty input", async () => {
    render(<Harness initial={["a", "b", "c"]} />);
    await userEvent.click(screen.getByRole("button", { name: "Remove tag b" }));
    expect(screen.getByTestId("value")).toHaveTextContent("a|c");
    await userEvent.type(screen.getByLabelText("Tags"), "{Backspace}");
    expect(screen.getByTestId("value")).toHaveTextContent("a");
  });

  it("suggests matching existing tags", async () => {
    render(<Harness suggestions={["goa", "goal", "beach"]} initial={["goal"]} />);
    await userEvent.type(screen.getByLabelText("Tags"), "go");
    const options = screen.getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["goa"]);
    await userEvent.click(options[0]);
    expect(screen.getByTestId("value")).toHaveTextContent("goal|goa");
  });
});
