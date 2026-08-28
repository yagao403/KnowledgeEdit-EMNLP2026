from __future__ import annotations
from collections import UserList
import json
import os
from pathlib import Path
import traceback
from typing_extensions import Self
from typing import Iterable, Callable, overload, SupportsIndex, Generic, TypeVar
from xml.etree import ElementTree as ET

from .message import STMessage, Message, Tag, BaseMessage, parse_tags, START, STOP, tag_to_sec, Recipient, SEC
from core import DATA_PATH
from core.metadata import Metadata
from core.utils import remove_comments


MsgT = TypeVar("MsgT", bound=BaseMessage)

class BaseMessages(UserList[MsgT], Generic[MsgT]):
    msg_class = BaseMessage
    xml_tag = "base_messages"

    def __init__(
        self,
        messages : Iterable[MsgT],
        metadata: dict | None = None,
        notes: str | None = None,
    ):
        super().__init__(messages)
        self.notes = notes
        self.metadata = Metadata(metadata or {})

    @property
    def contains_target(self) -> bool:
        return any(msg.contains_target for msg in self)

    def copy(self):
        return self.__class__(self, metadata=self.metadata, notes=self.notes)

    def __str__(self):
        s = f"<{self.xml_tag}>\n"
        for msg in self:
            s += str(msg) + "\n"
        s += f"</{self.xml_tag}>"
        return s

    def __repr__(self):
        return self.__str__()

    @overload
    def __getitem__(self, i: SupportsIndex) -> MsgT: ...
    @overload
    def __getitem__(self, i: slice) -> Self: ...

    def __getitem__(self, i: SupportsIndex | slice):
        item = super().__getitem__(i)
        if isinstance(i, slice):
            return self.__class__(
                item,  # type: ignore[call-arg]
                metadata=self.metadata,
                notes=self.notes,
            )
        return item

    @overload
    def __setitem__(self, i: SupportsIndex, value: MsgT | None) -> None: ...
    @overload
    def __setitem__(self, i: slice, value: Iterable[MsgT | None]) -> None: ...

    def __setitem__(self, i: SupportsIndex | slice, value):
        return super().__setitem__(i, value)

    def __add__(self, other: Iterable[MsgT]) -> Self:
        return self.__class__(
            list(self) + other,  # type: ignore
            metadata=self.metadata,
            notes=self.notes,
        )

    def __iadd__(self, other: Iterable[MsgT]) -> Self:
        if hasattr(other, "notes"):
            raise TypeError("Cannot add BaseMessages + BaseMessages")
        super().extend(other)
        return self

    def extend(self, other: Iterable[MsgT]):
        self.__iadd__(other)

    @classmethod
    def from_string(cls, xml_string: str) -> Self:
        element = ET.fromstring(xml_string)
        return cls.from_element(element)

    def to_string(self):
        name = "step_messages"
        s = f"<{name}>\n"

        element = self.to_element()
        for sub_element in element:
            s += ET.tostring(sub_element, encoding="unicode") + "\n\n"
        s += f"</{name}>"
        return s

    def to_element(self) -> ET.Element:
        element = ET.Element("step_messages")
        for msg in self:
            if msg is not None:
                element.append(msg.to_element())

        if self.notes:
            notes = ET.Element("notes")
            notes.text = self.notes
            element.append(notes)

        if self.metadata:
            metadata = ET.Element("metadata")
            metadata.text = self.metadata.to_json()
            element.append(metadata)

        return element

    @classmethod
    def from_element(cls, element: ET.Element,
                     new_format: bool = True, old_tags: bool = False) -> Self:
        messages = []
        notes = None
        metadata = None
        for sub_element in element:
            if sub_element.tag == "metadata":
                if metadata is not None:
                    raise SyntaxError(f"More than one metadata in {element}")
                if sub_element.text is not None:
                    metadata = Metadata.from_json(sub_element.text)

            elif sub_element.tag == "notes":
                if notes is not None:
                    raise SyntaxError(f"More than one notes in {element}")
                notes = sub_element.text

            else:
                messages.append(cls.msg_class.from_element(sub_element, new_format=new_format, old_tags=old_tags))

        return cls(messages, notes=notes, metadata=metadata)

    @classmethod
    def from_xml_path(cls, path: Path, new_format: bool = True) -> Self:
        msgs = cls.from_element(ET.parse(path).getroot(), new_format=new_format)
        msgs.metadata["path"] = path
        if True:
            # Re-tag action messages from old runs
            for msg in msgs:
                if Tag.IPYTHON in msg.all_tags and Tag.ACTION not in msg.tags:
                    msg.tags.add(Tag.ACTION)
                for section in msg.sections:
                    if hasattr(section, "content") and section.content.startswith("<thinking_summary>"):
                        section.tags |= {Tag.STEP_NOTES}
                    if hasattr(section, "tags") and Tag.THINKING_SUMMARY in section.tags:
                        # Replace THINKING_SUMMARY with STEP_NOTES
                        section.tags.remove(Tag.THINKING_SUMMARY)
                        section.tags.add(Tag.STEP_NOTES)

        return msgs

    def to_xml_path(self, path: os.PathLike):
        with open(path, 'w') as f:
            f.write(self.to_string())

    # @property
    # def contains_target(self) -> bool:
    #     return any(msg.contains_target for msg in self)

    @property
    def action(self) -> MsgT | None:
        """Get the last action message."""
        i = self.rfind(lambda m: Tag.ACTION in m.tags)
        # There should be no status message after the action
        status_idx = self.rfind(lambda m: Tag.STATUS in m.tags)
        return self[i] if i >= 0 and status_idx < i else None

    @property
    def output(self) -> MsgT | None:
        """Get the last output message."""
        i = self.rfind(lambda m: Tag.ACTION in m.tags)
        if i < 0:
            return None
        i_out = self[i:].find(lambda m: Tag.OUTPUT in m.tags)
        return self[i + i_out] if i_out >= 0 else None

    def has_exceptions(self) -> bool:
        output = self.output
        return output is not None and ("<exception#" in output.content)

    def get_action_section_by_tag(self, tag: Tag, single_step: bool = True) -> str | None:
        if single_step:
            action_idx = self.rfind(lambda m: Tag.ACTION in m.tags)
            if action_idx == -1:
                return None
            action_message = self[action_idx]
            action_section = None
            for sec in action_message.sections[::-1]:
                if tag in getattr(sec, "tags", {}):
                    action_section = sec
                    break
            if action_section is None:
                return None
            action_section = action_section.content
        else:
            section_idx = self.rfind(lambda m: tag in m.all_tags)
            if section_idx == -1:
                return None
            action_section = self[section_idx]
            action_section = action_section.content

        section_name = tag_to_sec(tag)
        action_section = parse_tags(action_section, tags=[section_name])[section_name]
        return action_section

    def get_step_notes(self, single_step: bool = True) -> str | None:
        step_notes = self.get_action_section_by_tag(tag=Tag.STEP_NOTES, single_step=single_step)
        return step_notes

    def get_ipython_cell(self, single_step: bool = True) -> str | None:
        ipython_cell = self.get_action_section_by_tag(tag=Tag.IPYTHON, single_step=single_step)
        return ipython_cell

    def get_workspace(self, before: bool) -> MsgT | None:
        i_action = self.rfind(lambda m: Tag.ACTION in m.tags)
        if i_action < 0:  # type: ignore
            return None
        if before:
            i = self[:i_action].find(lambda m: Tag.WORKSPACE in m.tags)
            return self[i] if i >= 0 else None
        else:
            i = self[i_action:].find(lambda m: Tag.WORKSPACE in m.tags)
            return self[i + i_action] if i >= 0 else None

    def hide_thinking(self, except_last: bool = True):
        """Hide thinking in all action messages except the last one."""
        msgs = []
        idx = self.rfind(lambda m: Tag.ACTION in m.tags)
        for i, msg in enumerate(self):
            if Tag.ACTION not in msg.tags:
                msgs.append(msg)
                continue

            if except_last and i == idx:
                msgs.append(msg)
            else:
                msgs.append(msg.without_tags({Tag.THINKING}))

        return self.__class__(msgs, metadata=self.metadata, notes=self.notes)

    def is_thinking_hidden(self, except_last: bool = True) -> bool:
        """Check if thinking is hidden in all action messages except the last one."""
        idx = self.rfind(lambda m: Tag.ACTION in m.tags)
        for i, msg in enumerate(self):
            if Tag.ACTION not in msg.tags:
                continue

            if except_last and i == idx:
                continue

            if Tag.THINKING in msg.all_tags:
                return False

        return True

    def keep_workspace(
        self,
        before: bool = True,  # Keep the workspace message before the last action
        after: bool = True,  # Keep the workspace message after the last action
        strict: bool = False,
    ):
        i_action = self.rfind(lambda m: Tag.ACTION in m.tags)
        if i_action < 0:
            if strict:
                raise ValueError("No action message found")
            return

        if not before:
            i = self[:i_action].rfind(lambda m: Tag.WORKSPACE in m.tags)
            if i >= 0:
                del self[i]
                i_action -= 1

        if not after:
            i = self[i_action:].find(lambda m: Tag.WORKSPACE in m.tags)
            if i >= 0:
                del self[i + i_action]

    def strip_output(self, strict: bool = False) -> Self:
        """Remove all messages after the last assistant message."""
        i = self.rfind(lambda m: m.role == 'assistant')
        if i < 0:
            if strict:
                raise ValueError("No assistant message found")
            return self.copy()

        return self[:i + 1]

    def strip_last_assistant_msg(self, strict: bool = False) -> Self:
        """Remove the last assistant message and all messages after it."""
        i = self.rfind(lambda m: m.role == 'assistant')
        if i < 0:
            if strict:
                raise ValueError("No assistant message found")
            return self.copy()

        return self[:i]

    def find(self, filter: Callable[[BaseMessage], bool]) -> int:
        """Find the index of the first message that matches the filter.
        If no message matches, return -1.
        """
        for i, msg in enumerate(self):
            if filter(msg):
                return i
        return -1

    def rfind(self, filter: Callable[[BaseMessage], bool]) -> int:
        """Find the index of the last message that matches the filter.
        If no message matches, return -1.
        """
        for i in range(len(self) - 1, -1, -1):
            if filter(self[i]):
                return i
        return -1

    def findall(self, filter: Callable[[BaseMessage], bool]) -> list[int]:
        """Find the indices of all messages that match the filter.
        If no message matches, return an empty list.
        """
        return [i for i, msg in enumerate(self) if filter(msg)]

    def strip_after_last_target(self) -> Self | None:
        """Remove all messages after the last message that contains a target. Return None if there is no target."""
        i = self.rfind(lambda m: m.contains_target)  # type: ignore
        if i < 0:
            return None
        return self[:i + 1]

    def remove_targets(self):
        """Remove all targets from the messages."""
        for msg in self:
            msg.remove_targets()

    def with_tags(self, tags: list[Tag]) -> Self | None:
        """Return copy that keeps messages with any of the `tags` plus sections with any of the `tags`."""
        new_messages = []
        for msg in self:
            if new_msg := msg.with_tags(tags):
                new_messages.append(new_msg)

        if new_messages:
            return self.__class__(new_messages, metadata=self.metadata, notes=self.notes)
        return None

    def without_tags(self, tags: list[Tag]) -> Self | None:
        """Return copy that removes messages with any of the `tags` and sections with any of the `tags`."""
        new_messages = []
        for msg in self:
            if new_msg := msg.without_tags(tags):
                new_messages.append(new_msg)

        if new_messages:
            return self.__class__(new_messages, metadata=self.metadata, notes=self.notes)
        return None

    def split_by_action(self, keep_workspace: bool = False) -> tuple[Self, MsgT]:
        """Split a step to a list of messages before the last action and the last action."""
        if keep_workspace:
            self.keep_workspace(before=True, after=False)
        i = self.rfind(lambda m: Tag.ACTION in m.tags)
        if i < 0:
            raise ValueError("No action message found")
        action = self[i]
        # assert action.contains_target or Tag.FAILED_PARSE_RESPONSE in action.tags, "The last action message must be a target or a failed parse response."
        return self[:i], action

    def add_label(self, label: str):
        """Add a label to the metadata."""
        self.metadata.setdefault("labels", [])
        self.metadata["labels"].append(label)

# TODO: Rename to STMessages
class STStepMessages(BaseMessages[STMessage]):
    """STStepMessages holds the messages of a single agent step.
    It behaves like list[STMessage] except it also holds a dictionary of notes."""
    msg_class = STMessage
    xml_tag = "step_messages"  # TODO: rename to "st_messages"

    def to_messages(self, teacher: bool = True, dropout_rate: float = 1.0) -> Messages:
        "Convert the messages to a list of messages."
        messages: list[Message] = []
        for st_msg in self:
            assert isinstance(st_msg, STMessage), f"Expected STMessage, got {type(st_msg)}"
            if msg := st_msg.to_message(teacher, dropout_rate):
                messages.append(msg)
        return Messages(messages, metadata=self.metadata, notes=self.notes)

    def apply_recipient_and_dropout(self, teacher: bool = True, dropout_rate: float = 1.0) -> STStepMessages:
        "Return STStepMessages where recipient and dropout has been applied to the messages."
        messages = self.to_messages(teacher, dropout_rate)
        st_messages = [msg.to_st_message() for msg in messages]
        return STStepMessages(st_messages, metadata=self.metadata, notes=self.notes)

    def apply_dropout(self, dropout_rate: float) -> STStepMessages:
        "Apply dropout to the messages."
        step = STStepMessages([
            msg.apply_dropout(dropout_rate) for msg in self
        ], notes=self.notes)
        return step

    def keep_recipients(self, recipients: list[Recipient]) -> STStepMessages:
        "Keep only messages with the given recipients."
        step = STStepMessages([
            msg.keep_recipients(recipients)
            for msg in self if msg.recipient in recipients
        ], notes=self.notes)
        return step


class Messages(BaseMessages[Message]):
    msg_class = Message
    xml_tag = "messages"

    def to_serialized(self) -> list[dict]:
        """Convert the messages to a list of dictionaries for serialization."""
        return [msg.to_serialized() for msg in self]


def get_steps_from_run(run_path: Path, load_config: bool = True, suffix: str = "") -> list[STStepMessages]:
    suffix = f"_{suffix}" if suffix else suffix
    step_paths = sorted([
        step_path
        for step_path in run_path.glob(f"[0-9][0-9][0-9]{suffix}.xml")
        if step_path.name != "000.xml" and step_path.parent.name != "messages"
    ])

    steps: list[STStepMessages] = []
    for step_path in step_paths:
        step_messages = STStepMessages.from_xml_path(step_path)
        if step_messages.metadata is None:
            step_messages.metadata = Metadata()
        step_messages.metadata.update({
            "path": to_placeholder_path(step_path),
        })
        if load_config:
            config_path = run_path / "config.json"
            if config_path.exists():
                step_messages.metadata["run_config"] = json.loads(config_path.read_text())
        steps.append(step_messages)

    return steps

def get_multiple_steps_from_xml(
    xml_path: Path,
    must_exist: bool = True,  # Raise error if file does not exist
    must_parse: bool = True,  # Raise error if parsing fails
    only_with_target: bool = False,  # Only return steps with target
    verbose: bool = False,
) -> list[STStepMessages]:

    if not (xml_path).exists():
        if must_exist:
            raise FileNotFoundError(f"{xml_path}: not found")
        if verbose:
            print(f"{xml_path}: not found")
        return []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        if must_parse:
            raise ValueError(f"Processing {xml_path} failed: {e}") from e

        if verbose:
            print(f"Processing {xml_path} failed: {e}")
        return []

    steps = []
    for i, element in enumerate(root.findall("step_messages")):
        try:
            # To load messages with old tags, set old_tags=True here
            step = STStepMessages.from_element(element, new_format=True, old_tags=False)
        except Exception as e:
            if must_parse:
                traceback.print_exc()
                raise ValueError(f"Processing exercise {i+1} from {xml_path} failed: {e}") from e

            else:
                if verbose:
                    print(f"Processing exercise {i+1} from {xml_path} failed: {e}")
        else:
            if not only_with_target or step.contains_target:
                steps.append(step)
    return steps

EXERCISES_TAG = "exercises"
def steps_to_exercise_xml(steps: list[STStepMessages], xml_path: Path):
    s = f"<{EXERCISES_TAG}>\n\n"
    for step in steps:
        if step is not None:
            s += step.to_string() + "\n\n"
    s += f"</{EXERCISES_TAG}>\n"
    with open(xml_path, "w") as f:
        f.write(s)


def to_placeholder_path(path: Path) -> str:
    s_path = str(path)
    s_path = s_path.replace(str(DATA_PATH), "DATA_PATH")
    return s_path


def resolve_placeholder_path(s_path: str | Path) -> Path:
    """Resolve a placeholder path to a real path."""
    s_path = str(s_path)
    if s_path.startswith("DATA_PATH"):
        s_path = s_path.replace("DATA_PATH", str(DATA_PATH), 1)
    return Path(s_path).resolve()


def read_steps_from_xml(
    xml_path: Path,
    only_with_target: bool = False,  # Only return steps with target
    verbose: bool = False,
) -> list[STStepMessages] | list[Messages]:
    """Read steps from an XML file.

    The difference to get_multiple_steps_from_xml:
    - never raises an error
    - returns either STStepMessages or Messages depending on the file format
    """
    if not (xml_path).exists():
        return []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        if verbose:
            print(f"Loading {xml_path} failed: {e}")
            traceback.print_exc()
        return []

    root_name = root.tag
    if root_name in ("step_messages", "st_messages"):
        cls = STStepMessages if root_name == "step_messages" else Messages
        # The file contains a single step
        steps = [cls.from_element(root, new_format=True)]

    elif root_name in ("exercises", "messages_collection"):
        # The file contains a collection of steps
        tag = "step_messages" if root_name == "exercises" else "messages"
        cls = STStepMessages if root_name == "exercises" else Messages
        steps = []
        for element in root.findall(tag):
            try:
                step = cls.from_element(element, new_format=True)
            except Exception as e:
                if verbose:
                    print(f"Loading step from {xml_path} failed: {e}")
                    traceback.print_exc()
            else:
                steps.append(step)
    else:
        if verbose:
            print(f"Unknown root element {root_name} in {xml_path}")
        return []

    if only_with_target:
        steps = [step for step in steps if step.contains_target]

    return steps  # type: ignore


def split_action(
    action: STMessage,
    strip_ipython: bool = True,
    strip_comments: bool = True,
) -> tuple[str, str]:
    thinking_sections = [s for s in action.sections if Tag.THINKING in s.tags]
    thinking = "".join(s.content for s in thinking_sections)

    ipython_cell = [s for s in action.sections if Tag.IPYTHON in s.tags]
    ipython_cell = "".join(s.content for s in ipython_cell)
    ipython_code = ipython_cell
    if strip_ipython:
        ipython_code = ipython_code.replace(START.ipython_cell, "").replace(STOP.ipython_cell, "")

    # print("Extracted IPython code:", repr(ipython_code))
    if strip_comments:
        assert strip_ipython, "strip must be True when no_comments is True"
        try:
            ipython_code = remove_comments(ipython_code, remove_empty_lines=True)
        except Exception as e:
            raise ValueError(f"Removing comments failed: {e}") from e

    if not ipython_code.endswith("\n"):
        ipython_code += "\n"

    return thinking, ipython_code
