"""纯函数的单元测试——不涉及网络、LLM、Qdrant。"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

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
    exclude_frontmatter: list[str] | None = None,
) -> AgenticObsidianVaultToQdrantHandler:
    # 绕过 __init__（它会构建 crew + Qdrant 客户端）；文件夹逻辑只需要
    # 下面这些属性。
    handler = AgenticObsidianVaultToQdrantHandler.__new__(
        AgenticObsidianVaultToQdrantHandler
    )
    handler._vault_root = Path("D:/vault")
    handler._include_folders = set(include or [])
    handler._exclude_folders = set(exclude or [])
    handler._exclude_frontmatter = set(exclude_frontmatter or [])
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
        # 如果剥掉标记后输出为空，则返回原文，而不是发送空邮件。
        assert self.strip("[来源1]") == "[来源1]"


class TestNormalizePath:
    """AgenticAutoReplyHandler._verify_sources 通过内部的 _normalize()
    把模型输出的路径与 Qdrant 中存储的路径做比较。模型输出可能是
    JSON 转义的（D:\\Self\\...）或 posix 风格的（D:/Self/...）。"""

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


class TestProcessedMessageIds:
    """Gmail history 会把同一封邮件的 messageAdded 事件重复投递，
    监听器必须按邮件 ID 去重，否则一封来信会被回复多次。"""

    def test_first_mark(self):
        state = GmailInboxState()
        assert state.mark_message_processed("msg1") is True
        assert state.is_message_processed("msg1") is True

    def test_duplicate_rejected(self):
        state = GmailInboxState()
        state.mark_message_processed("msg1")
        assert state.mark_message_processed("msg1") is False

    def test_unrelated_message_not_affected(self):
        state = GmailInboxState()
        state.mark_message_processed("msg1")
        assert state.is_message_processed("msg2") is False

    def test_survives_roundtrip(self):
        # 状态要能持久化到磁盘再加载，去重记录不能丢
        import tempfile

        state = GmailInboxState()
        state.mark_message_processed("msg1")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        state.save(path)
        loaded = GmailInboxState.load_state(path)
        assert loaded.is_message_processed("msg1") is True
        path.unlink()

    def test_remark_after_discard(self):
        # 失败路径会 discard 已处理标记（允许重试）；重试成功后必须能
        # 重新 mark（返回 True），否则 history 重复投递会绕过去重。
        state = GmailInboxState()
        state.mark_message_processed("msg1")
        state.processed_message_ids.discard("msg1")
        assert state.mark_message_processed("msg1") is True


class TestTrimProcessedMessageIds:
    """已处理集合只增不减会让状态文件无限膨胀，必须裁剪到上限。"""

    @staticmethod
    def trim(state: "GmailInboxState") -> None:
        # 直接实例化监听器会连带 GmailServiceAdapter 初始化；
        # 绕过 __init__ 挂上 _state 即可只测裁剪逻辑本身。
        from email_assistant.gmail.inbox import GmailInboxListener

        listener = object.__new__(GmailInboxListener)
        listener._state = state
        listener._trim_processed_message_ids()

    def test_under_limit_untouched(self):
        state = GmailInboxState()
        state.processed_message_ids = {"msg1"}
        self.trim(state)
        assert state.processed_message_ids == {"msg1"}

    def test_over_limit_trimmed(self):
        from email_assistant.gmail.inbox import PROCESSED_IDS_MAX

        state = GmailInboxState()
        state.processed_message_ids = {f"msg{i}" for i in range(PROCESSED_IDS_MAX + 500)}
        self.trim(state)
        assert len(state.processed_message_ids) == PROCESSED_IDS_MAX


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
        # 3 字节的 CJK 字符；截断在字符中间也不能抛异常
        text = "汉" * 5000
        result = storage._normalize_text(text)
        assert len(result.encode("utf-8")) <= QdrantStorage.MAX_LENGTH_BYTES
        assert isinstance(result, str)


class TestEmailResponseModel:
    def test_schema_descriptions_have_no_citation_marker_instructions(self):
        # schema 描述会经由 instructor 发给模型；它们不能要求在收件人
        # 可见的内容里带 [来源N] 标记。
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

    @pytest.mark.skipif(sys.platform != "win32", reason="反斜杠只在 Windows 才是路径分隔符")
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

    def test_special_folders_excluded_via_config(self):
        # .trash / .obsidian 等特殊文件夹没有内置排除逻辑，完全靠
        # .env 里的 OBSIDIAN_EXCLUDE_FOLDERS 配置生效。
        handler = _folder_handler(exclude=[".trash", ".obsidian"])
        assert handler._in_scope("D:/vault/.trash/a.md") is False
        assert handler._in_scope("D:/vault/.obsidian/a.md") is False


class TestExcludeFrontmatter:
    # 当配置的排除键与笔记的 frontmatter 键有交集时（如 Excalidraw 的
    # excalidraw-plugin），on_created 会跳过该文件。
    @staticmethod
    def excluded(handler, frontmatter: dict) -> bool:
        return bool(handler._exclude_frontmatter & frontmatter.keys())

    def test_excalidraw_key_excluded(self):
        handler = _folder_handler(exclude_frontmatter=["excalidraw-plugin"])
        assert self.excluded(handler, {"excalidraw-plugin": "raw"}) is True

    def test_other_frontmatter_proceeds(self):
        handler = _folder_handler(exclude_frontmatter=["excalidraw-plugin"])
        assert self.excluded(handler, {"tags": ["x"]}) is False

    def test_empty_config_proceeds(self):
        handler = _folder_handler()
        assert self.excluded(handler, {"excalidraw-plugin": ""}) is False
