import { test, expect } from "@playwright/test";

test.describe("App shell", () => {
  test("renders header with brand name", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".brand-text h1")).toBeVisible();
  });

  test("shows connection status indicator", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".conn")).toBeVisible();
  });

  test("renders legend", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".legend")).toBeVisible();
  });

  test("shows hint text", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".hint")).toBeVisible();
  });
});

test.describe("Side panel", () => {
  test("is hidden on load", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".side-panel")).not.toHaveClass(/side-panel--open/);
  });

  test("closes with Escape key when backend is available", async ({ page }) => {
    await page.goto("/");
    const errorBanner = page.locator(".overlay-banner.error");
    const hasError = await errorBanner.isVisible().catch(() => false);
    test.skip(hasError as boolean, "Backend not available in this environment");

    await page.waitForSelector("canvas", { state: "visible" });
    const canvas = page.locator("canvas").first();
    await canvas.click({ position: { x: 500, y: 400 } });
    await page.keyboard.press("Escape");
    await expect(page.locator(".side-panel")).not.toHaveClass(/side-panel--open/);
  });
});

test.describe("Hex data loading", () => {
  test("connects to backend or shows error banner", async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(3000);
    const connected = page.locator(".conn-connected");
    const errored = page.locator(".overlay-banner.error");
    const eitherVisible =
      (await connected.isVisible().catch(() => false)) ||
      (await errored.isVisible().catch(() => false));
    expect(eitherVisible).toBe(true);
  });
});
