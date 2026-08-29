"""Unit tests for pure functions — no network, no LLM, no Qdrant."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from email_assistant import models  # noqa: E402
from email_assistant.gmail.handlers import AgenticAutoReplyHandler  # noqa: E402
from email_assistant.gmail.inbox import GmailInboxState  # noqa: E402
from email_assistant.obsidian.handlers import (  # noqa: E402
    AgenticObsidianVaultToQdrantHandler,
)
from email_assistant.storage import QdrantStorage  # noqa: E402


def _folder_handler(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> AgenticObsidianVaultToQdrantHandler:
    # Bypass __init__ (which builds a crew + Qdrant client); the folder logic
    # only needs the three attributes set below.
    handler = AgenticObsidianVaultToQdrantHandler.__new__(
        AgenticObsidianVaultToQdrantHandler
    )
    handler._vault_root = Path("D:/vault")
    handler._include_folders = set(include or [])
    handler._exclude_folders = set(exclude or [])
    return handler


class TestStripCitationMarkers:
    strip = staticmethod(AgenticAutoReplyHandler._strip_citation_markers)

    def test_chinese_marker(self):
        assert self.strip("我的微信号是 blodguy_ink[来源1]，欢迎加我") == (
            "我的微信号是 blodguy_ink，欢迎加我"
        )

    def test_bare_index_markers(self):
        assert self.strip("My WeChat is ink[1] and QQ is 123[2].") == (
            "My WeChat is ink and QQ is 123."
        )

    def test_english_source_marker(self):
        assert self.strip("see (source 1) for details") == "see for details"

    def test_reference_marker(self):
        assert self.strip("ref [Reference 3] here") == "ref here"

    def test_marker_between_text(self):
        assert self.strip("价格是[3]100元") == "价格是100元"

    def test_no_markers_untouched(self):
        assert self.strip("没有任何引用的普通回复") == "没有任何引用的普通回复"

    def test_empty_and_none(self):
        assert self.strip("") == ""
        assert self.strip(None) is None

    def test_all_markers_stripped_falls_back_to_original(self):
        # If stripping would produce empty output, return the original instead
        # of sending an empty email.
        assert self.strip("[来源1]") == "[来源1]"


class TestNormalizePath:
    """AgenticAutoReplyHandler._verify_sources compares model-emitted paths
    against Qdrant-stored paths via its inner _normalize(). Model output can
    be JSON-escaped (D:\\Self\\...) or posix-style (D:/Self/...)."""

    @staticmethod
    def normalize(path: str) -> str:
        return path.replace("\\\\", "\\").replace("/", "\\")

    def test_json_escaped_matches_plain(self):
        stored = "D:\\Self Obsidian\\个人资料.md"
        emitted = "D:\\\\Self Obsidian\\\\个人资料.md"
        assert self.normalize(emitted) == self.normalize(stored)

    def test_posix_matches_windows(self):
        stored = "D:\\Self Obsidian\\个人资料.md"
        emitted = "D:/Self Obsidian/个人资料.md"
        assert self.normalize(emitted) == self.normalize(stored)


class TestUpdateLastHistoryId:
    def test_first_update(self):
        state = GmailInboxState()
        assert state.update_last_history_id(100) is True
        assert state.last_history_id == 100

    def test_forward_update(self):
        state = GmailInboxState(last_history_id=100)
        assert state.update_last_history_id(200) is True
        assert state.last_history_id == 200

    def test_stale_update_rejected(self):
        state = GmailInboxState(last_history_id=200)
        assert state.update_last_history_id(100) is False
        assert state.last_history_id == 200

    def test_duplicate_update_rejected(self):
        state = GmailInboxState(last_history_id=200)
        assert state.update_last_history_id(200) is False
        assert state.last_history_id == 200


class TestNormalizeText:
    def test_short_text_untouched(self):
        storage = QdrantStorage.__new__(QdrantStorage)
        assert storage._normalize_text("hello") == "hello"

    def test_long_text_truncated_to_limit(self):
        storage = QdrantStorage.__new__(QdrantStorage)
        long_text = "a" * (QdrantStorage.MAX_LENGTH_BYTES + 100)
        result = storage._normalize_text(long_text)
        assert len(result.encode("utf-8")) <= QdrantStorage.MAX_LENGTH_BYTES

    def test_multibyte_truncation_no_crash(self):
        storage = QdrantStorage.__new__(QdrantStorage)
        # 3-byte CJK chars; cut mid-character must not raise
        text = "汉" * 5000
        result = storage._normalize_text(text)
        assert len(result.encode("utf-8")) <= QdrantStorage.MAX_LENGTH_BYTES
        assert isinstance(result, str)


class TestEmailResponseModel:
    def test_schema_descriptions_have_no_citation_marker_instructions(self):
        # The schema descriptions are sent to the model via instructor; they
        # must not ask for [来源N] markers in the recipient-visible content.
        import json

        schema = json.dumps(models.EmailResponse.model_json_schema())
        assert "来源" not in schema
        assert "citation markers" not in schema or "without" in schema


class TestTopLevelFolder:
    def test_file_in_folder(self):
        handler = _folder_handler()
        assert handler._top_level_folder("D:/vault/实习经历/abc.md") == "实习经历"

    def test_file_at_vault_root(self):
        handler = _folder_handler()
        assert handler._top_level_folder("D:/vault/README.md") is None

    def test_file_outside_vault(self):
        handler = _folder_handler()
        assert handler._top_level_folder("E:/other/abc.md") is None

    def test_no_vault_root_configured(self):
        handler = _folder_handler()
        handler._vault_root = None
        assert handler._top_level_folder("D:/vault/实习经历/abc.md") is None

    def test_windows_separator_path(self):
        handler = _folder_handler()
        assert handler._top_level_folder("D:\\vault\\学习情况\\a.md") == "学习情况"


class TestInScope:
    def test_root_files_always_in_scope(self):
        handler = _folder_handler(include=["实习经历"])
        assert handler._in_scope("D:/vault/README.md") is True

    def test_include_list_restricts(self):
        handler = _folder_handler(include=["实习经历", "个人资料"])
        assert handler._in_scope("D:/vault/实习经历/a.md") is True
        assert handler._in_scope("D:/vault/学习情况/a.md") is False

    def test_exclude_list_removes(self):
        handler = _folder_handler(exclude=["日记"])
        assert handler._in_scope("D:/vault/日记/a.md") is False
        assert handler._in_scope("D:/vault/实习经历/a.md") is True

    def test_exclude_wins_over_include(self):
        handler = _folder_handler(include=["实习经历"], exclude=["实习经历"])
        assert handler._in_scope("D:/vault/实习经历/a.md") is False

    def test_empty_config_accepts_all(self):
        handler = _folder_handler()
        assert handler._in_scope("D:/vault/任意文件夹/a.md") is True
