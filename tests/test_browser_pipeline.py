"""The browser adapter must consume the original preprocessing/tracking code unchanged."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from courtvision.browser.runtime import BrowserPipeline
from courtvision.perception.student import OUTPUTS, StudentConfig, StudentDetector


def test_input_tensor_matches_desktop_and_resets_at_cuts():
    browser = BrowserPipeline()
    desktop = StudentDetector(StudentConfig())
    captured = []
    desktop.infer_arrays = lambda x, sizes: (captured.append(x.copy()) or [([], [])])
    rng = np.random.default_rng(7)
    # Odd native dimensions exercise the original OpenCV area resize and bottom/right padding.
    for i in range(4):
        image = rng.integers(0, 256, (71, 113, 3), dtype=np.uint8)
        if i == 3:
            browser.history.clear()
            desktop.reset()
        actual = browser.prepare(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).ravel(), 113, 71, i, i/29.97)
        desktop.detect(image, i, i/29.97)
        np.testing.assert_array_equal(actual, captured[-1][0])


def outputs():
    raw = {name: np.zeros((1 if name.endswith('hm') else 2, 136, 240), np.float32) for name in OUTPUTS}
    for name in ('ball_hm', 'person_hm'):
        raw[name].fill(-20)
    raw['person_hm'][0, 70, 100] = 2
    raw['person_wh'][:, 70, 100] = np.log([20, 45])
    # Three plausible raw ball peaks must never become three independent ball identities.
    raw['ball_hm'][0, 60, 102] = 2
    raw['ball_hm'][0, 40, 30] = 1
    raw['ball_hm'][0, 80, 200] = .5
    return raw


def test_shared_trackers_keep_ids_and_publish_only_one_ball():
    pipeline = BrowserPipeline()
    ids = []
    for index in range(4):
        pipeline.prepare(np.full((540, 960, 4), 100, np.uint8), 960, 540, index, index/29.97)
        result = json.loads(pipeline.finish(outputs()))
        people = [b for b in result['detections'] if b['class'] == 'person']
        balls = [b for b in result['detections'] if b['class'] == 'sports ball']
        assert len(people) == 1
        assert len(balls) == 1
        assert balls[0]['id'] == 'Ball'
        assert result['calibrated'] is False
        ids.append(people[0]['id'])
    assert len(set(ids)) == 1


def test_shipped_python_bundle_matches_original_sources():
    root = Path(__file__).resolve().parents[1] / 'courtvision'
    package = root / 'server/static/browser-python.zip'
    manifest = json.loads(package.with_suffix('.json').read_text())
    assert hashlib.sha256(package.read_bytes()).hexdigest() == manifest['sha256']
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == {'courtvision/' + name for name in manifest['sources']}
        for name, digest in manifest['sources'].items():
            source = (root / name).read_bytes()
            assert hashlib.sha256(source).hexdigest() == digest
            assert archive.read('courtvision/' + name) == source
