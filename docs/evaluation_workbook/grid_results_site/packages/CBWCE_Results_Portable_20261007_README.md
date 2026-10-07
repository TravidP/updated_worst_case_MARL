# CBWCE Results Viewer — Portable Edition

Protocol v7 evaluation results with an English-only interface. This package contains the complete current website and its exported CSV/JSON data. It works offline after extraction; it does not need Python, Node.js, SUMO, a training environment, or an internet connection when using the included native launchers.

## Quick start

1. Extract the **entire ZIP** to a normal local folder. Do not run a launcher inside the ZIP preview.
2. Start the launcher for your operating system below.
3. Your default browser opens automatically. Keep the server window open while viewing the site.
4. Press **Ctrl+C**, or close the server window, to stop the viewer.

The site runs only on `127.0.0.1`, on your own computer. It normally uses port `8878`; if that port is busy, the launcher selects another free port and prints the correct URL. If no browser opens, copy that printed URL into your browser. Files and folder names may contain spaces; the launchers do not depend on your current working directory.

### Windows (64-bit)

- **Double-click `Start-Windows.cmd`**. It automatically chooses the x64 or ARM64 executable.
- On an x64 computer, you can also **double-click `Start-CBWCE.exe`** directly.
- These are standalone local-server executables. No installer, Python installation, or administrator access is required.
- The executables are unsigned. If Windows displays a security notice, review the package source and checksums and follow your organization's policy. Do not disable security protections.

Optional Command Prompt usage:

```bat
Start-Windows.cmd --port 8880
Start-CBWCE.exe --no-browser --port 0
```

### macOS (Intel and Apple Silicon)

- Double-click **`Start-macOS.command`**. It selects the Intel or Apple Silicon binary and opens the site.
- If extraction did not preserve executable permissions, open Terminal in the extracted folder and run:

```sh
chmod +x Start-macOS.command
./Start-macOS.command
```

- The binaries are unsigned and not notarized. If macOS prevents execution, review the source and use the Python fallback below if allowed by your system policy. Do not disable system security protections.

### Linux (x86-64 and ARM64)

Open a terminal in the extracted folder and run:

```sh
bash Start-Linux.sh
```

For desktop double-click use, mark `Start-Linux.sh` executable and choose your file manager's option to run it in a terminal. The script selects the correct CPU binary automatically.

```sh
chmod +x Start-Linux.sh
./Start-Linux.sh --port 8880
```

The automatic browser opener uses `xdg-open`. On a minimal Linux installation, copy the printed URL into a browser manually instead. On a headless machine use `--no-browser`; the viewer still binds only to the local computer.

### Optional Python fallback (all platforms)

If you cannot use the native launchers, the package includes `server.py`, which uses only the Python standard library. Python is needed **only for this fallback or optional checksum verification**.

Windows:

```bat
py -3 server.py --port 8880
```

Linux / macOS:

```sh
python3 server.py --port 8880
```

Then open `http://127.0.0.1:8880/` yourself. Keep the terminal open and press Ctrl+C to stop. Do not open `dist/index.html` directly: the browser must load the CSVs through the local HTTP server.

## Using the viewer

- **Network:** choose Grid or Monaco.
- **Evaluate datasets:** both networks provide Seen and Test. Grid additionally provides **Hangzhou realworld data for grid**; Monaco additionally provides **Monaco real world dataset**. The other network's real-world dataset is hidden.
- **Controller:** select IA2C, MA2C, IQLL, PPO, or multiple controllers.
- **Scenario:** select a scenario within the current dataset.
- The overview displays the selected controller/method combinations. Colors identify methods; line styles identify controllers.
- Wheel zoom, zoom buttons, and horizontal dragging adjust a shared time window. Reset restores the full view. Smoothing changes only the displayed curves, not the stored measurements.
- The metrics table compares all five methods independently within each selected controller.
- Panels A and B have independent controller/method selections and share the time window and axes.
- **Download PNG** exports the current view and smoothing. Its legend is included below the plot, centered and automatically wrapped so it does not cover curves or get clipped. Each panel exports its own selected curves.
- **Download filtered CSV** exports raw time-series measurements for the current overview selection. Summary CSV and provenance/paired-comparison downloads are also available.
- The interface, chart labels, and downloaded PNG legends are English only. Original bilingual metadata remains in JSON files for data preservation.

## Included datasets and files

| Dataset | Time-series CSV files | Summary groups | Rollout records | Viewer availability |
| --- | ---: | ---: | ---: | --- |
| Main Grid, Seen + Test | 460 | 460 | 4,600 | Grid: Seen / Test |
| Main Monaco, Seen + Test | 460 | 460 | 4,600 | Monaco: Seen / Test |
| Hangzhou realworld data for grid | 20 | 20 | 200 | Grid only |
| Monaco real world dataset | 20 | 20 | 200 | Monaco only |
| Legacy Monaco replay, skipped OD | 20 | 20 | 200 | Archive only; hidden from viewer |

The package retains **all 980 time-series CSVs**, each with 3,600 data rows, plus all existing summary, rollout, paired-comparison and auxiliary CSVs. There are **993 CSV files in total**. The 9,600 currently selectable rollout records and 200 legacy records are included. These are display exports: per-group time series summarize ten runs; individual per-second raw rollout trajectories, model checkpoints, training environments and upstream simulation inputs are not part of this viewer package.

```text
Start-CBWCE.exe              Windows x64 standalone launcher
Start-Windows.cmd            Windows architecture-aware double-click entry
Start-Linux.sh               Linux entry (x86-64 / ARM64)
Start-macOS.command          macOS entry (Intel / Apple Silicon)
bin/                        Additional native launchers
server.py                   Optional Python standard-library fallback
verify_package.py           Optional integrity check
FILES_SHA256.json            SHA-256 checksums of every other packaged file
PACKAGE_INFO.json            Dataset inventory and validation counts
source/launcher.go          Full source of the native launcher
source/build_package.py     Reproducible package builder (run in original checkout)
source/README.md            Build instructions
dist/index.html             Website page
dist/app.js                 Viewer and PNG export program
dist/styles.css             Website styles
dist/data/                  Main Grid results and registries
  series/                   Grid time-series CSVs
  networks/monaco/          Main Monaco results and time-series CSVs
  supplementary/group12_v1/grid/
  supplementary/monaco_repaired_v1/monaco/
  supplementary/monaco_legacy_v1/monaco/  Preserved archival data
```

Catalogs, provenance, validation records and auxiliary files already present in `dist/` are copied without alteration. Nothing needs to be restored from a separate archive. The supplementary display names follow the requested names; naming does not establish verified upstream provenance. Consult the dataset catalogs and the viewer's result-status notes for provenance and experiment limitations.

## Check that extraction preserved all files

With Python 3.8 or later, run one of:

```bat
py -3 verify_package.py
```

```sh
python3 verify_package.py
```

The command checks every packaged program, website asset, CSV and JSON against `FILES_SHA256.json`. It reports missing or modified files. The manifest itself is excluded from its own checksum list. An adjacent `.zip.sha256` file is provided for verifying the ZIP before extraction. These checksums detect changed files; they are not a digital signature.

## Troubleshooting

- **Data missing / loading error:** extract the entire ZIP and run a launcher, not the HTML file. Check the package integrity and reload the browser.
- **Old interface:** confirm that you opened the URL printed by this package's launcher. A previous server on port 8878 may be a different copy. Refresh the page.
- **Port conflict:** the native launcher automatically chooses another port if 8878 is busy. For a specific port, pass `--port 8880`. An explicitly selected busy port produces an error.
- **Browser fails to open:** open the printed local URL manually.
- **Permission or architecture error:** use the correct OS entry script. On Linux/macOS it restores the selected binary's executable bit if necessary. Unsupported CPUs can use the Python fallback.
- **Downloads:** the browser controls the download destination and any save prompt. Data in the package is not overwritten by chart/CSV downloads.

## Validation and portability

The build checks all packaged checksums, CSV counts and time-series lengths, and verifies that every selectable catalog scenario has all controller/method series. The Linux x86-64 launcher is exercised with HTTP requests, including a port-conflict test. Windows and macOS executables are cross-compiled from the same standard-library Go source; their native formats are inspected, but they have not been executed on Windows/macOS in this build environment. No operating-system runtime was bundled or downloaded.
