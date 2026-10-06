# Cashew QC UI and Sample Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dense Cashew QC dashboard with a bright Vietnamese workflow UI, add a verified static `_yl` sample-image library, and use Three.js only as an optional explanatory visual.

**Architecture:** The FastAPI prediction contract remains unchanged. The frontend becomes a small set of static modules: a sample manifest and local sample images, an application module for input/state/result rendering, and a Three.js scene module that gracefully falls back to static HTML. A curation script runs the existing pipeline offline to create a reproducible twelve-image manifest before files are copied into `static/samples/`.

**Tech Stack:** FastAPI, existing `POST /predict`, vanilla HTML/CSS/JavaScript, Canvas 2D, Three.js (pinned browser module), Python/pytest, existing YOLO + TorchScript pipeline.

**Spec:** `docs/superpowers/specs/2026-10-06-cashew-qc-redesign-design.md`

## Global Constraints

- Preserve model weights, class labels, detection thresholds, and the existing `GET /health`, `GET /models`, and `POST /predict?model=<id>` contract.
- Default UI language is Vietnamese and default theme is light.
- Ship exactly 12 local static samples: two verified images for each of `tb`, `loai1`, `loai2`, `loai3`, `lbw`, and `bad_output`.
- Select samples only from `*_yl` source folders, with `badoutput_yl` mapping to classifier class `bad_output`.
- A sample must have at least one detection and a highest-confidence detection matching its expected class; manifest records the observed confidence and count.
- The primary result UI must never depend on WebGL, a remote dependency, or animation.
- Three.js must honor `prefers-reduced-motion` and render no accessible information that is unavailable as HTML.
- Do not add a framework, database, new prediction endpoint, or server-side sample-copy operation.

## Review Focus

- A sample manifest points to a missing image or maps `badoutput_yl` to the wrong public class; Task 2 validates all files and allowed classes.
- A sample thumbnail is treated as a trusted prediction without calling the actual model; Task 4 routes samples through the same preview-and-predict path as uploaded files.
- API downtime leaves the user with only an ambiguous “offline” badge; Task 4 provides explanatory copy and a retry action.
- A no-detection response crashes summary calculations or presents an empty chart as success; Task 5 renders a dedicated no-detection state.
- WebGL or motion preference failure breaks image input/result controls; Task 3 tests the static fallback and keeps all controls outside the scene.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/curate_sample_images.py` | Deterministically scan `_yl` sources, run `CashewPipeline`, select and copy two verified images per class, and write the manifest. |
| `static/samples/manifest.json` | Runtime-read-only list of the twelve shipped samples and their verification metadata. |
| `static/samples/*.png` | The twelve optimized selected image files used by the deployed webapp. |
| `static/vendor/three.module.min.js` | Pinned local Three.js ESM build; avoids making core UI depend on CDN availability. |
| `static/js/cashew-scene.js` | Isolated optional Three.js cashew/processing scene with a `destroy()` method and static fallback signal. |
| `static/app.js` | UI state, API calls, sample library, previews, result summaries, Canvas annotation, detail table, health/model states. |
| `static/index.html` | Semantic layout, style tokens, responsive design, progressive enhancement entry points. |
| `tests/test_sample_curation.py` | Unit tests for source-class mapping, deterministic candidate selection, manifest schema, and asset references without loading real model weights. |
| `tests/test_static_webapp.py` | Static contract tests for light-theme default, sample-manifest loading, same analysis path, retry state, reduced-motion fallback, and Three.js local module reference. |

### Task 1: Add sample curation primitives

**Files:**
- Create: `scripts/curate_sample_images.py`
- Create: `tests/test_sample_curation.py`

**Interfaces:**
- Consumes: source directories at project root and `CashewPipeline.predict(image) -> list[dict]`.
- Produces: `class_source_map() -> dict[str, Path]`, `evaluate_candidate(path, expected_class, pipeline) -> dict | None`, `select_samples(records, per_class=2) -> list[dict]`, and `write_manifest(samples, output_dir) -> Path`.

- [ ] **Step 1: Write the failing mapping and candidate-selection tests**

```python
def test_class_source_map_uses_only_yl_folders(tmp_path):
    sources = class_source_map(tmp_path)
    assert set(sources) == {"tb", "loai1", "loai2", "loai3", "lbw", "bad_output"}
    assert sources["bad_output"].name == "badoutput_yl"

def test_candidate_requires_matching_top_prediction(tmp_path):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"fake")
    pipeline = FakePipeline([{"class": "loai1", "conf_class": 0.97}])
    assert evaluate_candidate(image_path, "loai2", pipeline) is None

def test_select_samples_returns_two_distinct_records_per_class():
    selected = select_samples(make_records(), per_class=2)
    assert Counter(item["expected_class"] for item in selected) == {name: 2 for name in CLASSES}
```

- [ ] **Step 2: Run the curation tests to verify failure**

Run: `python -m pytest tests/test_sample_curation.py -v`

Expected: FAIL because the curation module and its public functions do not exist.

- [ ] **Step 3: Implement the deterministic curation module**

```python
CLASS_SOURCE_DIRS = {
    "tb": "tb_yl", "loai1": "loai1_yl", "loai2": "loai2_yl",
    "loai3": "loai3_yl", "lbw": "lbw_yl", "bad_output": "badoutput_yl",
}

def evaluate_candidate(path: Path, expected_class: str, pipeline) -> dict | None:
    image = cv2.imread(str(path))
    predictions = pipeline.predict(image) if image is not None else []
    matching = [p for p in predictions if p["class"] == expected_class]
    if not matching:
        return None
    top = max(matching, key=lambda prediction: prediction["conf_class"])
    return {"source": path, "expected_class": expected_class,
            "detected_count": len(predictions), "predicted_class": top["class"],
            "confidence": round(float(top["conf_class"]), 4)}
```

Sort input paths by name, rank valid candidates by confidence descending then filename ascending, and copy only selected files into the requested output folder with stable names such as `loai1-01.png`. Generate JSON containing `id`, `path`, `expected_class`, `display_name`, `detected_count`, `predicted_class`, and `confidence`.

- [ ] **Step 4: Run the curation tests to verify success**

Run: `python -m pytest tests/test_sample_curation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the curation primitives**

```powershell
git add scripts/curate_sample_images.py tests/test_sample_curation.py
git commit -m "feat: add verified sample image curation"
```

### Task 2: Generate and validate deployed sample assets

**Files:**
- Create: `static/samples/manifest.json`
- Create: `static/samples/tb-01.png`, `static/samples/tb-02.png`, `static/samples/loai1-01.png`, `static/samples/loai1-02.png`
- Create: `static/samples/loai2-01.png`, `static/samples/loai2-02.png`, `static/samples/loai3-01.png`, `static/samples/loai3-02.png`
- Create: `static/samples/lbw-01.png`, `static/samples/lbw-02.png`, `static/samples/bad-output-01.png`, `static/samples/bad-output-02.png`
- Modify: `tests/test_sample_curation.py`

**Interfaces:**
- Consumes: curation CLI `python scripts/curate_sample_images.py --output static/samples --per-class 2`.
- Produces: a manifest where each `path` is relative to `/samples/` and each referenced asset exists beneath `static/samples/`.

- [ ] **Step 1: Add failing shipped-asset tests**

```python
def test_shipped_manifest_has_two_verified_assets_per_class():
    manifest = json.loads((ROOT / "static/samples/manifest.json").read_text(encoding="utf-8"))
    assert Counter(item["expected_class"] for item in manifest["samples"]) == {
        "tb": 2, "loai1": 2, "loai2": 2, "loai3": 2, "lbw": 2, "bad_output": 2,
    }
    for item in manifest["samples"]:
        assert item["expected_class"] == item["predicted_class"]
        assert (ROOT / "static" / item["path"].lstrip("/")).is_file()
```

- [ ] **Step 2: Run the asset test to verify failure**

Run: `python -m pytest tests/test_sample_curation.py::test_shipped_manifest_has_two_verified_assets_per_class -v`

Expected: FAIL because no manifest or assets exist.

- [ ] **Step 3: Run curation against the installed model and select assets**

```powershell
python scripts/curate_sample_images.py --output static/samples --per-class 2 --model efficientnet_b3
```

Inspect the generated manifest and the twelve thumbnails. If a class lacks two matching results, report the class and candidate results instead of substituting incorrectly labeled images.

- [ ] **Step 4: Run the asset tests to verify success**

Run: `python -m pytest tests/test_sample_curation.py -v`

Expected: PASS and exactly twelve referenced sample files.

- [ ] **Step 5: Commit deployed samples and manifest**

```powershell
git add static/samples tests/test_sample_curation.py
git commit -m "feat: add verified Cashew QC sample images"
```

### Task 3: Build the optional local Three.js scene

**Files:**
- Create: `static/vendor/three.module.min.js`
- Create: `static/js/cashew-scene.js`
- Create: `tests/test_static_webapp.py`

**Interfaces:**
- Consumes: `#cashew-scene` host element and `window.matchMedia('(prefers-reduced-motion: reduce)')`.
- Produces: `createCashewScene(host: HTMLElement): { destroy(): void, active: boolean }` and a `data-scene-state` of `active`, `reduced`, or `fallback` on the host.

- [ ] **Step 1: Write the failing static-scene tests**

```python
def test_scene_uses_pinned_local_three_module():
    scene = (ROOT / "static/js/cashew-scene.js").read_text(encoding="utf-8")
    assert "../vendor/three.module.min.js" in scene
    assert "prefers-reduced-motion: reduce" in scene
    assert "destroy()" in scene

def test_webapp_has_non_webgl_scene_fallback():
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    assert 'id="cashew-scene"' in html
    assert "scene-fallback" in html
```

- [ ] **Step 2: Run the scene tests to verify failure**

Run: `python -m pytest tests/test_static_webapp.py -k scene -v`

Expected: FAIL because there is no local Three.js module or scene host.

- [ ] **Step 3: Implement the scene and fallback**

Vendor a pinned Three.js release into `static/vendor/three.module.min.js`. In `cashew-scene.js`, build a low-poly cashew from two rounded meshes, use a static orthographic camera on reduced motion, animate only with requestAnimationFrame otherwise, and call `renderer.dispose()` plus geometry/material disposal from `destroy()`. Set fallback state when WebGL renderer construction fails.

- [ ] **Step 4: Run the scene tests to verify success**

Run: `python -m pytest tests/test_static_webapp.py -k scene -v`

Expected: PASS.

- [ ] **Step 5: Commit the progressive-enhancement scene**

```powershell
git add static/vendor/three.module.min.js static/js/cashew-scene.js tests/test_static_webapp.py
git commit -m "feat: add accessible Three.js cashew scene"
```

### Task 4: Replace the input and health workflow UI

**Files:**
- Modify: `static/index.html`
- Create: `static/app.js`
- Modify: `tests/test_static_webapp.py`

**Interfaces:**
- Consumes: `/health -> {status, device, cuda, gpu_name, model}`, `/models -> {models, default}`, `/samples/manifest.json`, local file or sample URL.
- Produces: `selectInput({ file, source, sample })`, `runAnalysis()`, `refreshHealth()`, and `setConnectionState('ready'|'loading'|'offline'|'analyzing')`.

- [ ] **Step 1: Write failing workflow contract tests**

```python
def test_webapp_defaults_to_light_vietnamese_workflow():
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    assert 'lang="vi"' in html
    assert 'data-theme="light"' in html
    assert "Chọn ảnh" in html and "Phân tích ảnh" in html

def test_samples_share_the_upload_analysis_path():
    app = (ROOT / "static/app.js").read_text(encoding="utf-8")
    assert "selectInput" in app
    assert "runAnalysis" in app
    assert "fetch('/samples/manifest.json')" in app
    assert "fetch(`/predict${modelParam}`" in app

def test_offline_state_has_retry_action():
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    assert "Thử lại" in html
    assert 'id="retryConnection"' in html
```

- [ ] **Step 2: Run the workflow tests to verify failure**

Run: `python -m pytest tests/test_static_webapp.py -k 'workflow or samples or offline' -v`

Expected: FAIL because the existing page defaults to dark terminal UI and has no sample library/retry action.

- [ ] **Step 3: Implement semantic, responsive input UI and client state**

Replace the three-column dense dashboard with header, hero/input section, sample grid, image workspace, summary cards, result distribution, collapsible technical section, and expandable detail table. Keep `<canvas id="mainCanvas">` as the annotated image surface. Use button labels `Chọn ảnh`, `Dùng ảnh mẫu`, `Phân tích ảnh`, `Xem chi tiết`, and `Thử lại`.

Load `manifest.json`, construct sample cards from DOM APIs rather than string interpolation, fetch a chosen sample into a `File`, and pass it through `selectInput` exactly as uploads do. Selecting input previews only; `runAnalysis` alone posts `FormData` to `/predict`.

Map health failures to a visible message `Không kết nối được máy chủ phân tích` and wire `#retryConnection` to both `refreshHealth` and the model-list refresh. Disable analysis until a valid preview is selected and backend health is ready.

- [ ] **Step 4: Run workflow tests to verify success**

Run: `python -m pytest tests/test_static_webapp.py -k 'workflow or samples or offline' -v`

Expected: PASS.

- [ ] **Step 5: Manually verify responsive input flow**

Run: `python -m uvicorn api:app --host 127.0.0.1 --port 8000`

Open `http://127.0.0.1:8000`, verify light default, choose a local image, choose a sample, turn the backend off and use retry, then test at 375 px and 1440 px widths.

- [ ] **Step 6: Commit input workflow**

```powershell
git add static/index.html static/app.js tests/test_static_webapp.py
git commit -m "feat: redesign Cashew QC input workflow"
```

### Task 5: Render accessible analysis results and technical detail

**Files:**
- Modify: `static/app.js`
- Modify: `static/index.html`
- Modify: `tests/test_static_webapp.py`

**Interfaces:**
- Consumes: `/predict` payload `{ latency_ms, image_size, count, model_used, predictions }` and each prediction `{ bbox, class, conf_detect, conf_class, probs }`.
- Produces: `renderResults(data)`, `renderNoDetections()`, `drawPredictions()`, `openDetectionDetail(index)`, and `getQualitySummary(predictions)`.

- [ ] **Step 1: Write failing result-state tests**

```python
def test_summary_counts_are_safe_for_zero_predictions():
    app = (ROOT / "static/app.js").read_text(encoding="utf-8")
    assert "function getQualitySummary(predictions)" in app
    assert "if (!predictions.length)" in app
    assert "Không phát hiện hạt điều" in app

def test_result_ui_keeps_technical_metrics_collapsible():
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    assert "<details" in html
    assert "Thông tin kỹ thuật" in html
    assert "Độ tin cậy YOLO" in html
```

- [ ] **Step 2: Run result tests to verify failure**

Run: `python -m pytest tests/test_static_webapp.py -k 'summary or technical' -v`

Expected: FAIL because result state is spread across terminal-style chips and does not have a no-detection card.

- [ ] **Step 3: Implement result rendering and annotation**

Use the API payload directly. Show total detections, most common class, accepted count (`tb`, `loai1`, `loai2`), review/reject count (`loai3`, `lbw`, `bad_output`), and latency. Draw labelled boxes on the image canvas with class labels plus percentages. Keep color paired with plain-language labels and counts.

When `count === 0`, preserve the selected image, clear old boxes/table values, show `Không phát hiện hạt điều trong ảnh này`, and provide `Chọn ảnh khác`. Make the per-detection table optional; rows open a detail panel that shows the crop, classifier probability distribution, and YOLO confidence.

- [ ] **Step 4: Run result tests to verify success**

Run: `python -m pytest tests/test_static_webapp.py -k 'summary or technical' -v`

Expected: PASS.

- [ ] **Step 5: Run regression tests and perform model-backed smoke tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_sample_curation.py tests/test_static_webapp.py -v`

Expected: PASS.

With Uvicorn running, select one static sample from each class and confirm response count, predicted class, summary cards, rendered box labels, and opened detection detail match the API payload.

- [ ] **Step 6: Commit result UI**

```powershell
git add static/index.html static/app.js tests/test_static_webapp.py
git commit -m "feat: add clear Cashew QC analysis results"
```

### Task 6: Final browser, accessibility, and asset verification

**Files:**
- Modify if needed: `static/index.html`, `static/app.js`, `static/js/cashew-scene.js`, `tests/test_static_webapp.py`

**Interfaces:**
- Consumes: complete static app and the deployed FastAPI service.
- Produces: verified desktop/mobile/keyboard/reduced-motion behavior; no API contract changes.

- [ ] **Step 1: Add regression tests for deployment invariants**

```python
def test_no_runtime_cdn_is_required_for_core_assets():
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    assert "three.module.min.js" not in html
    assert "type=\"module\" src=\"/app.js\"" in html

def test_manifest_and_scene_are_served_from_static_tree():
    assert (ROOT / "static/samples/manifest.json").is_file()
    assert (ROOT / "static/vendor/three.module.min.js").is_file()
```

- [ ] **Step 2: Run final static tests**

Run: `python -m pytest tests/test_static_webapp.py tests/test_sample_curation.py -v`

Expected: PASS.

- [ ] **Step 3: Perform browser checks**

Verify keyboard upload activation, visible focus, status announcements, contrast, 375 px layout, 1440 px layout, reduced-motion behavior, forced WebGL fallback, and successful model analysis on one uploaded image plus one sample.

- [ ] **Step 4: Run full targeted verification**

Run: `python -m pytest tests/test_pipeline.py tests/test_model_eval.py tests/test_sample_curation.py tests/test_static_webapp.py -v`

Expected: all runnable tests PASS; existing model-eval skip remains acceptable when its dataset is absent.

- [ ] **Step 5: Commit final verification adjustments**

```powershell
git add static tests scripts
git commit -m "test: verify accessible Cashew QC webapp"
```
