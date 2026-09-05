async (page) => {
  const checks = [];
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const check = (condition, name) => {
    if (!condition) throw new Error(name);
    checks.push(name);
  };
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("http://127.0.0.1:8000/");
  await page.locator("#stage-loading").waitFor({ state: "hidden" });
  await page.waitForFunction(
    () => document.querySelector("#metric-success").textContent === "93.8%",
  );
  check(
    (await page.locator("#checkpoint option").count()) === 5,
    "five curated attempts",
  );
  check(
    (await page.locator("#checkpoint").inputValue()) === "showcase-ppo",
    "banana champion default",
  );
  check(
    (await page.locator("#timeline").inputValue()) === "0",
    "reduced motion disables autoplay",
  );
  await page.locator("#play-button").click();
  await page.waitForFunction(
    () => Number(document.querySelector("#timeline").value) > 0,
  );
  await page.getByRole("button", { name: "Pause replay", exact: true }).click();
  const paused = await page.locator("#timeline").inputValue();
  await page.waitForTimeout(550);
  check(
    (await page.locator("#timeline").inputValue()) === paused,
    "pause holds frame",
  );
  await page.locator("#timeline").fill("27");
  check(
    (await page.locator("#outcome").textContent()) === "Clean getaway",
    "scrub to terminal outcome",
  );
  await page.locator("#restart-button").click();
  await page.getByRole("button", { name: "Pause replay", exact: true }).click();
  check(
    Number(await page.locator("#timeline").inputValue()) <= 1,
    "restart resets frame",
  );
  await page.locator("#speed").selectOption("4");
  await page.locator("#play-button").click();
  await page.waitForFunction(
    () => Number(document.querySelector("#timeline").value) > 3,
  );
  await page.getByRole("button", { name: "Pause replay", exact: true }).click();
  checks.push("4x playback advances");
  await page.locator("#view-agent").click();
  check(
    (await page.locator("#view-agent").getAttribute("aria-pressed")) === "true",
    "Joyce visibility toggle",
  );
  await page.locator("#view-full").click();
  await page.locator("#compare-button").click();
  await page.locator("#comparison").waitFor({ state: "visible" });
  check(
    (await page.locator("#comparison-note").textContent()).includes(
      "Matched map",
    ),
    "comparison matches stage seed and grid",
  );
  check(
    (await page.locator("#timeline").getAttribute("max")) === "96",
    "comparison includes longer earlier attempt",
  );
  await page.locator("#timeline").fill("96");
  check(
    (await page.locator("#outcome").textContent()) === "Clean getaway",
    "long comparison preserves champion terminal state",
  );
  await page.locator("#close-comparison").click();
  check(
    (await page.locator("#timeline").getAttribute("max")) === "27",
    "closing comparison restores episode bounds",
  );
  await page.locator("#checkpoint").selectOption("showcase-random");
  await page.waitForFunction(
    () =>
      document.querySelector("#policy-badge").textContent === "Random policy",
  );
  await page.locator("#checkpoint").selectOption("showcase-bc");
  await page.waitForFunction(
    () => document.querySelector("#policy-badge").textContent === "Imitation",
  );
  await page.locator("#checkpoint").selectOption("scripted-door-0");
  await page.waitForFunction(
    () =>
      document.querySelector("#policy-badge").textContent === "Scripted guide",
  );
  checks.push("random imitation and scripted labels");
  await page.locator("#run-select").selectOption("camera-study-seed23");
  check(
    (await page.locator("#metric-success").textContent()) === "0.0%",
    "camera failure visible in test metric",
  );
  await page.locator("#run-select").selectOption("banana-refined-seed11");
  check(
    (await page.locator("#learning-chart circle").count()) === 18,
    "evaluation-only curve point count",
  );
  await page.locator("#validate-map").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#editor-status")
      .textContent.includes("geometric route"),
  );
  await page.getByRole("button", { name: "Floor", exact: true }).click();
  await page
    .getByRole("button", { name: "Banana, column 8, row 2", exact: true })
    .click();
  await page.locator("#validate-map").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#editor-status")
      .textContent.includes("exactly one banana"),
  );
  checks.push("editor rejects missing banana");
  await page.locator("#reset-map").click();
  await page.getByRole("button", { name: "Banana", exact: true }).click();
  await page
    .getByRole("button", { name: "Floor, column 7, row 2", exact: true })
    .focus();
  await page.keyboard.press("Enter");
  check(
    (await page.locator('#editor-board [data-tile="B"]').count()) === 1,
    "keyboard painting relocates unique banana",
  );
  await page.locator("#reset-map").click();
  await page.getByRole("button", { name: "Floor", exact: true }).click();
  await page
    .getByRole("button", { name: "Wall, column 1, row 1", exact: true })
    .click();
  check(
    (await page
      .getByRole("button", { name: "Wall, column 1, row 1", exact: true })
      .count()) === 1,
    "editor preserves boundary",
  );
  await page.locator("#run-map").click();
  await page.waitForFunction(() =>
    document.querySelector("#mode-label").textContent.includes("NEW ATTEMPT"),
  );
  check(
    (await page.locator("#policy-badge").textContent()) === "Transformer",
    "custom map uses real learned policy",
  );
  check(
    await page.locator("#run-map").isEnabled(),
    "inference restores button",
  );
  await page.route("**/api/play", (route) =>
    route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({ detail: "no model selected for this check" }),
    }),
  );
  await page.locator("#run-map").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#editor-status")
      .textContent.includes("no model selected"),
  );
  check(
    await page.locator("#run-map").isEnabled(),
    "missing-model error is recoverable",
  );
  await page.unroute("**/api/play");
  await page.route("**/api/validate", (route) => route.abort());
  await page.locator("#validate-map").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#editor-status")
      .textContent.includes("local Python server"),
  );
  checks.push("server-unavailable message");
  await page.unroute("**/api/validate");
  for (const width of [1440, 390, 360]) {
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 844 });
    await page.waitForTimeout(200);
    const layout = await page.evaluate(() => ({
      width: innerWidth,
      scroll: document.documentElement.scrollWidth,
      chart: document
        .querySelector("#learning-chart svg")
        .getAttribute("viewBox"),
    }));
    check(layout.scroll <= layout.width, `no overflow at ${width}px`);
    check(
      Number(layout.chart.split(" ")[2]) <= Math.max(330, width),
      `responsive chart at ${width}px`,
    );
  }
  check(errors.length === 0, "no uncaught browser errors");
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.reload();
  await page.locator("#stage-loading").waitFor({ state: "hidden" });
  return { passed: checks.length, checks, errors };
};
