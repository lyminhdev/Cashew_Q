from pathlib import Path


ROOT = Path(__file__).parent.parent


def test_webapp_defaults_to_light_vietnamese_workflow():
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    assert 'lang="vi"' in html
    assert 'data-theme="light"' in html
    assert "Chọn ảnh" in html
    assert "Phân tích ảnh" in html


def test_samples_share_the_upload_analysis_path():
    app = (ROOT / "static/app.js").read_text(encoding="utf-8")
    assert "function selectInput" in app
    assert "function runAnalysis" in app
    assert "fetch('/samples/manifest.json')" in app
    assert "fetch(`/predict${modelParam}`" in app


def test_offline_state_has_retry_action():
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    assert "Thử lại" in html
    assert 'id="retryConnection"' in html


def test_result_state_handles_no_detections_and_technical_detail():
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    app = (ROOT / "static/app.js").read_text(encoding="utf-8")
    assert "<details" in html
    assert "Thông tin kỹ thuật" in html
    assert "function getQualitySummary(predictions)" in app
    assert "Không phát hiện hạt điều" in app


def test_research_page_and_loading_steps_are_present():
    research = (ROOT / "static/research.html").read_text(encoding="utf-8")
    app = (ROOT / "static/app.js").read_text(encoding="utf-8")
    assert 'id="researchPage"' in research
    assert "94,84%" in research
    assert "77,84%" in research
    assert "Đang phát hiện hạt" in app
    assert "Đang phân loại từng hạt" in app
    assert "Ảnh ${number}" in app
