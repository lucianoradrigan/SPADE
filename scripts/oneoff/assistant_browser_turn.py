import sys, time
from playwright.sync_api import sync_playwright
BASE, SHOT = sys.argv[1], sys.argv[2]
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_context(viewport={"width": 1440, "height": 1300}).new_page()
    pg.goto(BASE + "/assistant", wait_until="domcontentloaded"); pg.wait_for_selector("[data-testid='stChatInputTextArea'], textarea", timeout=90000); pg.wait_for_timeout(3000)
    box = pg.locator("[data-testid='stChatInputTextArea'], textarea").first
    box.fill("Simulate the stock DC motor with a PI controller at 100 rad/s for 0.5 s."); box.press("Enter")
    t0 = time.time(); n = 0
    while time.time() - t0 < 180:
        n = pg.locator("[data-testid='stChatMessage']").count()
        if n >= 3: break
        pg.wait_for_timeout(1000)
    print(f"messages={n} after {time.time() - t0:.0f} s")
    pg.wait_for_timeout(3000); pg.screenshot(path=SHOT); b.close()
