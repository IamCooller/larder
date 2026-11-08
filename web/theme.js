(function () {
  "use strict";
  // Resolve the stored theme before the first paint so the page never flashes the wrong colours.
  var setting = "system";
  try { setting = localStorage.getItem("chat.theme") || "system"; } catch (e) {}
  if (["system", "light", "dark"].indexOf(setting) < 0) setting = "system";
  var dark = setting === "dark" ||
    (setting === "system" && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  document.documentElement.dataset.themeSetting = setting;
  var color = document.querySelector('meta[name="theme-color"]');
  if (color) color.setAttribute("content", dark ? "#141416" : "#ffffff");
})();
