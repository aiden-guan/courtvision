"""RF-DETR must never download pretrained weights without an explicit opt-in."""

import argparse

import pytest

from courtvision import cli
from courtvision.config import load_config
from courtvision.perception.rfdetr_adapter import (
    PretrainedDownloadNotAllowed,
    RFDETRConfig,
    RFDETRDetector,
    check_weights_available,
)


def test_no_weights_refuses_before_importing_rfdetr(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "rfdetr":
            raise AssertionError("rfdetr must not be imported when the download is refused")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(PretrainedDownloadNotAllowed, match="allow_pretrained_download"):
        RFDETRDetector(RFDETRConfig())._torch_model()


def test_missing_weights_file_is_refused(tmp_path):
    with pytest.raises(PretrainedDownloadNotAllowed, match="not found"):
        check_weights_available(RFDETRConfig(weights=str(tmp_path / "nope.pth")))


def test_existing_weights_file_and_explicit_opt_in_pass(tmp_path):
    weights = tmp_path / "fine_tuned.pth"
    weights.write_bytes(b"x")
    check_weights_available(RFDETRConfig(weights=str(weights)))
    check_weights_available(RFDETRConfig(allow_pretrained_download=True))
    check_weights_available(RFDETRConfig(backend="coreml"))  # CoreML packages are local by construction


def test_cli_refuses_auto_rfdetr_on_real_video_without_weights():
    cfg = load_config(None, [])
    with pytest.raises(cli.CLIError, match="--allow-download"):
        cli._check_rfdetr_weights(cfg, None, real_video=True)
    # sim renders resolve "auto" to colour blobs, so nothing to check
    cli._check_rfdetr_weights(cfg, None, real_video=False)


def test_cli_allow_download_flag_sets_the_opt_in():
    args = argparse.Namespace(allow_download=True)
    flags = cli._perception_flags(args)
    cfg = load_config(None, flags)
    assert cfg.perception.rfdetr.allow_pretrained_download is True
    cli._check_rfdetr_weights(cfg, None, real_video=True)


def test_parser_exposes_allow_download_on_run_and_serve():
    parser = cli.build_parser()
    assert parser.parse_args(["run", "--video", "x.mp4", "--allow-download"]).allow_download
    assert parser.parse_args(["serve", "--allow-download"]).allow_download
