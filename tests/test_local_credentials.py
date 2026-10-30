"""Local key persistence without shell execution or changes to unrelated settings."""
import stat

import pytest

from courtvision.credentials import runtime_env, save_jev_key


def test_key_roundtrip_preserves_settings_and_uses_private_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# keep this\nANTHROPIC_API_KEY="other-test-key"\nTYPESAFE_API_KEY=old-test-key\n')
    save_jev_key('test-key-$()`literal', path)
    assert path.read_text().startswith('# keep this\nANTHROPIC_API_KEY="other-test-key"\n')
    assert path.read_text().count("TYPESAFE_API_KEY=") == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert runtime_env(path, {}) == {"TYPESAFE_API_KEY": 'test-key-$()`literal'}
    assert runtime_env(path, {"TYPESAFE_API_KEY": "exported-test-key"})["TYPESAFE_API_KEY"] == "exported-test-key"
    save_jev_key("rotated-test-key", path)
    assert runtime_env(path, {})["TYPESAFE_API_KEY"] == "rotated-test-key"


@pytest.mark.parametrize("key", ["", "  ", "export TYPESAFE_API_KEY=test", "test\nkey", "test\x00key"])
def test_invalid_input_never_changes_file(tmp_path, key):
    path = tmp_path / ".env"
    path.write_text("UNCHANGED=value\n")
    with pytest.raises(ValueError):
        save_jev_key(key, path)
    assert path.read_text() == "UNCHANGED=value\n"


def test_dotenv_is_data_not_executable_code(tmp_path):
    path = tmp_path / ".env"
    path.write_text('export TYPESAFE_API_KEY=literal-$(no-command) # comment\nOTHER=value\n')
    assert runtime_env(path, {}) == {"TYPESAFE_API_KEY": "literal-$(no-command)"}
    path.write_text('TYPESAFE_API_KEY="unclosed\n')
    assert runtime_env(path, {}) == {}
