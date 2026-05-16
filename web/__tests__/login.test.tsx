import { render, screen } from "@testing-library/react";
import LoginPage from "@/app/login/page";

jest.mock("next/navigation", () => ({
  usePathname: () => "/login",
}));

describe("Login page", () => {
  it("renders the Sign in card title", () => {
    render(<LoginPage />);
    expect(screen.getByText("Sign in")).toBeInTheDocument();
  });

  it("shows auth coming soon description", () => {
    render(<LoginPage />);
    expect(screen.getByText(/coming in a future release/i)).toBeInTheDocument();
  });

  it("renders disabled sign-in buttons", () => {
    render(<LoginPage />);
    const buttons = screen.getAllByRole("button");
    buttons.forEach((btn) => expect(btn).toBeDisabled());
  });
});
