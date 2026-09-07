#!/usr/bin/env node
// `npx ragmcp <args>` runs the Python `rag` CLI through uv, pinned to this
// package's version. RAGMCP_EXTRAS=full,ocr adds optional extras;
// RAGMCP_UVX_FROM=<wheel or spec> overrides the source (used by the install tests).
"use strict"

const { spawnSync } = require("node:child_process")
const { version } = require("../package.json")

const extras = process.env.RAGMCP_EXTRAS ? `[${process.env.RAGMCP_EXTRAS}]` : ""
const spec = process.env.RAGMCP_UVX_FROM || `ragmcp${extras}==${version}`
const args = ["--from", spec, "rag", ...process.argv.slice(2)]

function launch(command, prefix) {
  return spawnSync(command, [...prefix, ...args], { stdio: "inherit" })
}

let result = launch("uvx", [])
if (result.error && result.error.code === "ENOENT") result = launch("uv", ["tool", "run"])

if (result.error && result.error.code === "ENOENT") {
  process.stderr.write(
    [
      "ragmcp needs uv, the Python package runner, which was not found on PATH.",
      "Install it once, then run this command again:",
      "  macOS / Linux : curl -LsSf https://astral.sh/uv/install.sh | sh",
      "  Windows       : powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"",
      "  any platform  : pip install uv",
      "",
    ].join("\n"),
  )
  process.exit(1)
}
if (result.error) {
  process.stderr.write(`${result.error.message}\n`)
  process.exit(1)
}
process.exit(result.status === null ? 1 : result.status)
