from __future__ import annotations

from typing import Any, Sequence
import base64
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from pydantic import BaseModel
import random
import re
from typing import Literal, get_args, Iterable, NamedTuple, Type
from typing_extensions import Self


import xml.etree.ElementTree as ET

from core.utils import xml_decoder, xml_encoder, Colors, MyEnum


Recipient = Literal[
    "teacher", "student",
    "student_dropout",  # teacher: always see, student: dropout
    "only_student_dropout",  # teacher: never see, student: dropout
    None,  # Teacher and student both see
]
Role = Literal["system", "assistant", "user", "none"]
ResponseFormat = Literal["inner_monologue", "ipython_cell"]

class Tag(MyEnum):
    BRIEFING = "briefing"
    GUIDELINES = "guidelines"
    MY_MEMORIES = "my_memories"
    TOOL_DOCS = "tool_docs"
    INIT_SCRIPT = "init_script"
    MONOLOGUE_INSTRUCTION = "monologue_instruction"
    STEP_NOTES_INSTRUCTION = "step_notes_instruction"
    IPYTHON_INSTRUCTION = "ipython_instruction"
    WORKSPACE = "workspace"
    STATUS = "status"
    THINKING = "thinking"

    TRUNCATED = "truncated"  # generation of the message was truncated

    # These tags are used when messages are rendered in the history
    # Note: step files contain only messages that are rendered in the history
    HIDDEN = "hidden"

    FAILED_PARSE_RESPONSE = "failed_parse_response"  # Agent output could not be parsed
    ACTION = "action"
    MONOLOGUE = "monologue"
    THINKING_SUMMARY = "th_summary"  # Deprecated, use STEP_NOTES
    STEP_NOTES = "step_notes"
    IPYTHON = "ipython"
    OUTPUT = "output"
    FEEDBACK = "feedback"
    REVIEWER_FEEDBACK = "reviewer_feedback"
    CAUSE_AND_EFFECT = "cause_and_effect_analysis"  # Step-by-step prediction, first step of anomaly detection
    PREDICTION = "prediction"  # Reorganizing predictions, second step of anomaly detection
    PREDICTION_CATEGORIZATION = "prediction_categorization"  # Categorization of predictions, third step of anomaly detection
    ANOMALY_SUMMARY = "anomaly"  # Summary of anomalies, fourth step of anomaly detection
    COMPLETION = "completion"  # unclassified completion message
    INJECTION = "injection"  # injected message (not from the LLM)
    NEW_LINES = "new_lines"  # new lines section for separating sections

    LOGPROB = "logprob"  # request logprobs

    CE = "ce"  # cross-entropy loss
    KL = "kl"  # KL divergence loss

class TrajectorySectionNames:
    think = "think"
    monologue = "inner_monologue"  # Used in multi-step actions
    # thinking_summary = "thinking_summary"  # Deprecated, use step_notes
    step_notes = "step_notes"
    ipython_cell = "ipython_cell"
    cause_and_effect = "cause_and_effect_analysis"
    prediction = "prediction"
    prediction_categorization = "prediction_categorization"
    anomaly = "anomaly"

SEC = TrajectorySectionNames()

class StartSequences:
    monologue: str = f"<{SEC.monologue}>\n"
    think: str = f"<{SEC.think}>\n"
    step_notes: str = f"<{SEC.step_notes}>\n"
    ipython_cell: str = f"<{SEC.ipython_cell}>\n"
    # candidate_cell: str = "<candidate_cell>"
    candidate_cell: str = "# candidate cell #"
    cause_and_effect: str = f"<{SEC.cause_and_effect}>\n"
    prediction: str = f"<{SEC.prediction}>\n"
    prediction_categorization: str = f"<{SEC.prediction_categorization}>\n"
    anomaly: str = "<discrepancies>\n"

class StopSequences:
    monologue: str = f"</{SEC.monologue}>"
    think: str = f"</{SEC.think}>"
    step_notes: str = f"</{SEC.step_notes}>"
    ipython_cell: str = f"</{SEC.ipython_cell}>"
    # candidate_cell: str = "</candidate_cell>"
    candidate_cell: str = "# end candidate cell #"
    cause_and_effect: str = f"</{SEC.cause_and_effect}>"
    prediction: str = f"</{SEC.prediction}>"
    prediction_categorization: str = f"</{SEC.prediction_categorization}>"
    anomaly: str = "</discrepancies>"

START = StartSequences()
STOP = StopSequences()


def get_section_identifiers(section_name: str, start_seq: str, stop_seq: str) -> dict:
    return {
        "section_name": section_name,
        "start_seq": start_seq,
        "stop_seq": stop_seq
    }

TAG_TO_SECTION_IDENTIFIERS = {
    Tag.THINKING: get_section_identifiers(SEC.think, START.think, STOP.think),
    Tag.MONOLOGUE: get_section_identifiers(SEC.monologue, START.monologue, STOP.monologue),
    Tag.IPYTHON: get_section_identifiers(SEC.ipython_cell, START.ipython_cell, STOP.ipython_cell),
    Tag.STEP_NOTES: get_section_identifiers(SEC.step_notes, START.step_notes, STOP.step_notes),
    Tag.CAUSE_AND_EFFECT: get_section_identifiers(SEC.cause_and_effect, START.cause_and_effect, STOP.cause_and_effect),
    Tag.PREDICTION: get_section_identifiers(SEC.prediction, START.prediction, STOP.prediction),
    Tag.PREDICTION_CATEGORIZATION: get_section_identifiers(SEC.prediction_categorization, START.prediction_categorization, STOP.prediction_categorization),
    Tag.ANOMALY_SUMMARY: get_section_identifiers(SEC.anomaly, START.anomaly, STOP.anomaly)
}


class ContentTarget(NamedTuple):
    content: str
    is_target: bool

class BackgroundColors:
    default = "\033[49m"
    old = "\033[48;2;250;218;221m"
    target = "\033[48;2;244;255;224m"
    edit = "\033[48;2;191;255;128m"

BG = BackgroundColors()

def get_pprint_color(role: Role, tags: set[Tag], target: bool) -> str:
    if role == "assistant":
        return Colors.BLUE
    if Tag.WORKSPACE in tags:
        return Colors.MAGENTA
    elif Tag.STATUS in tags:
        return Colors.GREEN
    elif Tag.OUTPUT in tags:
        return Colors.YELLOW
    elif Tag.FEEDBACK in tags or Tag.REVIEWER_FEEDBACK in tags:
        return Colors.RED
    else:
        return Colors.DEFAULT


def tags_to_str(tags: set[Tag]) -> str:
    return " ".join([t.value for t in tags])

def tags_from_str(tag_str: str, strict: bool = False) -> set[Tag]:
    if strict:
        return {Tag(t) for t in tag_str.split()}

    tags = set()
    for t in tag_str.split():
        try:
            tags.add(Tag(t))
        except ValueError:
            pass
    return tags

def tag_to_sec(tag: Tag) -> str:
    if tag in TAG_TO_SECTION_IDENTIFIERS.keys():
        return TAG_TO_SECTION_IDENTIFIERS[tag]["section_name"]
    else:
        raise ValueError(f'The tag "{tag}" is not associated with a section name.')

def tag_to_start_seq(tag: Tag) -> str:
    if tag in TAG_TO_SECTION_IDENTIFIERS.keys():
        return TAG_TO_SECTION_IDENTIFIERS[tag]["start_seq"]
    else:
        raise ValueError(f'The tag "{tag}" is not associated with a start sequence identifier.')

def tag_to_stop_seq(tag: Tag) -> str:
    if tag in TAG_TO_SECTION_IDENTIFIERS.keys():
        return TAG_TO_SECTION_IDENTIFIERS[tag]["stop_seq"]
    else:
        raise ValueError(f'The tag "{tag}" is not associated with a stop sequence identifier.')

def tags_from_element(element: ET.Element) -> set[Tag]:
    return tags_from_str(element.attrib.get("tags", ""))

def drop(recipient: Recipient, render_for_teacher: bool = True, dropout_rate: float | None = None) -> bool:
    "Determine if the message should be dropped."

    if not render_for_teacher and dropout_rate is None:
        raise ValueError("dropout_rate must be provided for student rendering")

    if recipient is None:
        return False

    target = "teacher" if render_for_teacher else "student"
    if recipient == target:
        return False

    if recipient == "student_dropout":
        if target != "teacher" and dropout_rate is None:
            raise ValueError("dropout_rate must be provided for student dropout")
        if target == "teacher" or random.random() > dropout_rate:  # type: ignore
            return False
    elif recipient == "only_student_dropout":
        # Teacher never sees this content
        if target == "teacher":
            return True
        # For student, apply dropout with the given rate
        if dropout_rate is None:
            raise ValueError("dropout_rate must be provided for student dropout")
        if random.random() > dropout_rate:  # keep for student with prob 1 - rate
            return False
    return True

class Image:
    encoded: str  # base64 encoded image string
    path: Path | None = None
    target: bool = False
    tags: set[Tag] = field(default_factory=set)

    def __init__(self, encoded: str, path: Path | None = None):
        self.encoded = encoded
        self.path = path

    @property
    def content(self) -> str:
        raise AttributeError("Image does not have content attribute.")

    @classmethod
    def from_path(cls, path: Path) -> Image:
        encoded = cls.encode(path)
        return cls(encoded, path)

    @staticmethod
    def encode(path: Path) -> str:
        """Encode the image to base64."""
        if not path.exists():
            raise FileNotFoundError(f"Image file {path} does not exist.")
        format = path.suffix[1:].lower()
        with open(path, "rb") as image_file:
            encoding = base64.b64encode(image_file.read()).decode('utf-8')

        return f"data:image/{format};base64,{encoding}"

    def to_element(self, parent: ET.Element | None = None, truncate: int | None = None): # truncate = None means no truncation
        if parent is None:
            element = ET.Element("image")
        else:
            element = ET.SubElement(parent, "image")
        if path := self.path:
            element.set("path", str(path))
        if truncate and len(self.encoded) > truncate:
            element.text = self.encoded[:truncate] + '...'
        else:
            element.text = self.encoded
        return element

    @classmethod
    def from_element(cls, element: ET.Element) -> Image:
        assert element.tag == "image", f"Invalid tag: {element.tag}"
        path = element.attrib.get("path", None)
        assert path is not None, "Image path is required"
        path = Path(path)
        encoded = cls.encode(path)
        return cls(encoded, path)

    def to_dict(self) -> dict[str, str]:
        return {
            "type": "image",
            "image": self.encoded,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Image:
        assert data["type"] == "image", f"Invalid type: {data['type']}"
        assert "image" in data and data["image"].startswith("data:image/"), "Image data must be base64 encoded"

        return cls(
            encoded=data["image"],
            path=None,  # Path is not stored in the dict representation
        )

    def __str__(self):
        element = self.to_element()
        # Shorten the encoded string for display
        if len(self.encoded) > 50:
            element.text = self.encoded[:50] + "..."
        else:
            element.text = self.encoded
        return ET.tostring(element, encoding="unicode")

@dataclass
class Section:
    content: str
    target: bool = False
    tags: set[Tag] = field(default_factory=set)
    recipient: Recipient = None
    tag: str = "s"

    @classmethod
    def empty(cls):
        return cls(content="")

    def __bool__(self):
        return bool(self.content)

    def __post_init__(self):
        assert self.recipient in get_args(Recipient), f"Invalid recipient: {self.recipient}"
        if self.recipient:
            assert not self.target, f"target is not allowed with recipient ({self.recipient})"

    def copy(self, **kwargs) -> Section:
        return replace(self, **kwargs)

    def to_element(self, parent: ET.Element | None = None) -> ET.Element:
        element = ET.Element(self.tag) if parent is None else ET.SubElement(parent, self.tag)
        if self.recipient:
            element.set("recipient", self.recipient)
        if self.target:
            element.set("target", "true")

        if isinstance(self.content, str):
            element.text = xml_encoder(self.content)
        else:
            self.content.to_element(element)
        if self.tags:
            element.set("tags", tags_to_str(self.tags))

        return element

    @classmethod
    def from_element(cls, element: ET.Element, new_format: bool = True) -> Section:
        if not new_format:
            return cls.from_element_old(element)

        assert element.tag == cls.tag, f"Invalid tag: {element.tag}"
        if len(element) == 1 and element[0].tag == "image":
            content = Image.from_element(element[0])
        else:
            content = xml_decoder(element.text) if element.text else ""

        return cls(
            content=content,
            recipient=element.attrib.get("recipient", None),  # type: ignore
            target=element.attrib.get("target", "false") == "true",
            tags=tags_from_element(element),
        )

    @classmethod
    def from_element_old(cls, element: ET.Element) -> Section:
        tags = tags_from_element(element)
        content = xml_decoder(element.text) if element.text else ""
        if element.tag == "section":
            return cls(content, tags=tags)
        elif element.tag == "target":
            return cls(content, target=True, tags=tags)
        else:
            assert element.tag in get_args(Recipient), f"Unknown section tag: {element.tag}"
            return cls(content, recipient=element.tag, tags=tags)  # type: ignore

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any]
        if isinstance(self.content, str):
            d = {
                "type": "text",
                "text": self.content,
                "tags": tags_to_str(self.tags),
            }
        else:
            d = {
                "type": "image",
                "image": self.content.to_dict(),
            }
        d["target"] = self.target
        return d

    def render(self, teacher: bool = True, dropout_rate: float | None = None) -> Section:
        "Render the message for the teacher or student."
        if drop(self.recipient, teacher, dropout_rate):
            return Section.empty()
        else:
            return Section(self.content, self.target, self.tags)

    def apply_dropout(self, dropout_rate: float) -> Section | str:
        if self.recipient == "student_dropout":
            if random.random() > dropout_rate:
                return self.copy(recipient=None)
            else:
                return self.copy(recipient="teacher")
        if self.recipient == "only_student_dropout":
            # Keep for student only with probability 1 - rate, otherwise drop entirely
            if random.random() > dropout_rate:
                return self.copy(recipient="student")
            else:
                return Section.empty()
        return self

    def __str__(self):
        xml = ET.tostring(self.to_element(), encoding="unicode")
        # If this section is marked as target, wrap with the target edit color
        if getattr(self, "target", False):
            return BG.target + xml + BG.default
        return xml

    def __repr__(self):
        s = f"{self.__class__.__name__}({repr(self.content)}"
        args = {}
        if self.recipient:
            args["recipient"] = repr(self.recipient)
        if self.target:
            args["target"] = self.target
        if self.tags:
            args["tags"] = repr(tags_to_str(self.tags))
        if args:
            s += ", "
            s += ", ".join(f"{k}={v}" for k, v in args.items())
        s += ")"
        return s

    @classmethod
    def wrapped_content(
        cls,
        content: str,
        name: str,
        end: str | None = None,
        **kwargs,
    ) -> Section:
        end = end or name
        return cls(
            f"--- START OF {name} ---\n" + content + f"\n--- END OF {end} ---\n",
            **kwargs,
        )

    @classmethod
    def format_request(
        cls,
        model: Type[BaseModel],
        **kwargs,
    ) -> Section:
        """Return a prompt instructing the model to match the given Pydantic schema."""
        # Handle both Pydantic v1 and v2
        if hasattr(model, "model_json_schema"):
            schema = model.model_json_schema()
        else:
            schema = model.schema()

        schema_str = json.dumps(schema, indent=2)
        return cls(
            "\n----\nReturn the result as a JSON object that matches the following schema:\n"
            f"{schema_str}",
            **kwargs,
        )

S = Section

SectionType = Section | Image


class BaseMessage:
    role: Role
    sections: list[SectionType]
    tags: set[Tag]

    recipient: Recipient = None
    target: bool = False
    eot_target: bool | None = None  # None means that the footer's target is the
                                    # same as the last section's target

    def __init__(
        self,
        role: Role,
        sections: str | Sequence[SectionType | str],
        *,
        recipient: Recipient = None,
        tags: set[Tag] | None = None,
        target: bool = False,
        eot_target: bool | None = None,
        strict: bool = True,  # Whether to check the roles
    ):
        if strict:
            assert role in get_args(Role), f"Invalid role: {role}"
        self.role = role

        if isinstance(sections, str):
            sections = [sections]

        # Remove empty sections and convert strings to Section
        self.sections = [
            section if not isinstance(section, str) else Section(section)
            for section in sections
            if section
        ]
        self.recipient = recipient
        self.tags = set(tags or ())

        if target and (
            any(getattr(section, "target", False) for section in sections)
            or eot_target
        ):
            raise ValueError("target sections should not be used when the whole message is a target")
        self.target = target
        self.eot_target = eot_target

    @property
    def contains_target(self) -> bool:
        return any(getattr(section, "target", False) for section in self.sections) or self.target

    def contains_tag(self, tag: Tag) -> bool:
        return any(getattr(section, "tags", set()) & {tag} for section in self.sections) or tag in self.tags

    def remove_targets(self):
        """Remove the target from the message."""
        self.target = False
        for section in self.sections:
            if hasattr(section, "target"):
                section.target = False

    @property
    def all_tags(self) -> set[Tag]:
        """Return all tags, including those in sections."""
        return set().union(self.tags, *(getattr(s, "tags", set()) for s in self.sections))

    @property
    def finished(self) -> bool:
        """Return True if the message is finished (no truncation tag)."""
        return len(self.sections) > 0 and Tag.TRUNCATED not in self.sections[-1].tags

    @finished.setter
    def finished(self, value: bool):
        if not self.sections:
            return
        if value:
            self.sections[-1].tags.discard(Tag.TRUNCATED)
        else:
            self.sections[-1].tags.add(Tag.TRUNCATED)

    @classmethod
    def empty(cls):
        return cls(role="none", sections=[])

    def __bool__(self):
        return self.role != "none" or bool(self.sections)

    def tags_in_sections(self) -> bool:
        return any(getattr(section, "tags", False) for section in self.sections)

    def copy(self, **kwargs):
        # Merge current attributes with overrides
        cls = self.__class__
        kwargs = {
            key: getattr(self, key)
            for key in cls.__annotations__.keys()
        } | kwargs

        # Create a new instance of the actual class
        return cls(**kwargs)

    @property
    def content(self) -> str:
        """Return the content of the message as a single string."""
        strings = []
        for section in self.sections:
            if isinstance(section, str):
                strings.append(section)
            elif isinstance(section, Section):
                strings.append(str(section.content))
            elif isinstance(section, Image):
                strings.append(f"[Image: {section.path}]")
            else:
                raise TypeError(f"Unknown section type: {type(section)}")

        return "".join(strings)

    @classmethod
    def from_string(cls, xml_string: str, new_format: bool = True):
        element = ET.fromstring(xml_string)
        return cls.from_element(element, new_format=new_format)

    def __str__(self):
        if not self:
            return "<empty/>"
        text_color = get_pprint_color(self.role, self.tags, self.target)
        xml = ET.tostring(self.to_element(), encoding="unicode")
        s = text_color + xml + Colors.DEFAULT

        # Highlight any section elements with target="true" using the target edit color
        # and then restore the outer color so the rest of the message retains its color.
        if self.target:
            return BG.target + s + BG.default

        try:
            pattern = re.compile(r'(<s\b[^>]*\btarget="true"[^>]*>.*?</s>)', re.DOTALL)
            s = pattern.sub(lambda m: BG.target + m.group(1) + BG.default, s)
        except Exception:
            pass

        if self.eot_target or (self.eot_target is None and self.sections and getattr(self.sections[-1], "target", False)):
            try:
                closing_pattern = re.compile(fr'(</{self.role}>)')
                s = closing_pattern.sub(lambda m: BG.target + m.group(1) + BG.default, s, count=1)
            except Exception:
                pass
        return s

    def __repr__(self):
        return str(self)

    def for_review(self) -> str:
        show_role = Tag.OUTPUT not in self.tags

        s = f"<{self.role}>" if show_role else ""
        s += self.content
        s += f"</{self.role}>" if show_role else ""
        return s

    def to_element(self, parent: ET.Element | None = None) -> ET.Element:
        tag = self.role
        if parent is None:
            element = ET.Element(tag)
        else:
            element = ET.SubElement(parent, tag)
        if self.tags:
            element.set("tags", tags_to_str(self.tags))
        if self.recipient:
            element.set("recipient", self.recipient)
        if self.target:
            element.set("target", "true")
        if self.eot_target is not None:
            element.set("eot_target", "true" if self.eot_target else "false")
        # if len(self.sections) == 1 and isinstance(self.sections[0], str):
        #     element.text = xml_encoder(self.sections[0])
        for section in self.sections:
            if not hasattr(section, "to_element"):
                section = Section(section)  # type: ignore
            if isinstance(section, Image):
                section.to_element(element, truncate=50) # type: ignore
            else:
                section.to_element(element) # type: ignore

        return element

    @classmethod
    def from_element(cls, element: ET.Element,
                     new_format: bool = True, old_tags: bool = False) -> Self:

        # Backward compatibility for response_format
        if "response_format" in element.attrib:
            response_format = element.attrib.pop("response_format")
            return cls.from_element_with_response_format(
                element, response_format, new_format=new_format, old_tags=old_tags
            )

        if len(element):
            sections = []
            if element.text:
                sections.append(xml_decoder(element.text))
            for sub_element in element:
                if sub_element.tag == "image":
                    sections.append(Image.from_element(sub_element))
                elif sub_element.tag == "short_content":
                    pass  # Ignore short_content for backward compatibility
                else:
                    if new_format:
                        sections.append(Section.from_element(sub_element))
                    else:
                        sections.append(Section.from_element_old(sub_element))

                if sub_element.tail:
                    sections.append(xml_decoder(sub_element.tail))
        else:
            sections = xml_decoder(element.text) if element.text else ""

        kwargs = {}
        if "recipient" in element.attrib:
            kwargs["recipient"] = element.attrib.get("recipient")
        if "target" in element.attrib:
            kwargs["target"] = element.attrib.get("target") == "true"
        if "eot_target" in element.attrib:
            kwargs["eot_target"] = element.attrib.get("eot_target") == "true"
        return cls(
            element.tag,  # type: ignore
            sections,
            tags=tags_from_element(element),
            **kwargs,
        )

    @classmethod
    def from_element_with_response_format(
        cls,
        element: ET.Element,
        response_format: str,
        new_format: bool = True,
        old_tags: bool = False,  # <inner_monologue> or <run_ipython> tags
    ) -> Self:
        assert "response_format" not in element.attrib
        msg = cls.from_element(element, new_format=new_format)

        for i, section in enumerate(msg.sections):
            if isinstance(section, str):
                msg.sections[i] = Section(section, target=msg.target)
            # for sections with targets, we keep the target of the section
            elif hasattr(section, "target"):
                section.target = msg.target or section.target # type: ignore

        if old_tags and response_format == "run_ipython":
            start = "<run_ipython>\n"
            stop = "</run_ipython>"
        elif old_tags and response_format == "inner_monologue":
            start = "<inner_monologue>\n"
            stop = "</inner_monologue>"
        else:
            start = START.monologue if response_format == "inner_monologue" else START.ipython_cell
            stop = STOP.monologue if response_format == "inner_monologue" else STOP.ipython_cell

        msg.sections = (
            [Section(start, target=False)]  # We force the opening tag
            + msg.sections
        )
        msg.sections[-1].content += stop  # type: ignore
        msg.target = False

        return msg

    def without_tags(self, tags: Iterable[Tag]) -> Self | None:
        """Remove the message or any sections that have any of the specified tags. Return None if nothing remains."""
        # If any of the tags are in the whole message, return None.
        if any(tag in self.tags for tag in tags):
            return None
        sections = [
            sec for sec in self.sections
            if not any(tag in sec.tags for tag in tags)
        ]
        if sections:
            return self.copy(sections=sections)
        return None

    def with_tags(self, tags: Iterable[Tag]) -> Self | None:
        """Only keep sections with the given tags. Return None if no sections remain or no tags are defined."""
        # If any of the tags are in the whole message, return the whole message.
        if any(tag in self.tags for tag in tags):
            return self
        if self.tags_in_sections():
            # Return a message with the sections that have any of the specified tags.
            sections = []
            for sec in self.sections:
                has_tag = False
                if getattr(sec, "tags", False):
                    assert isinstance(sec, Section), sec
                    for tag in tags:
                        if tag in sec.tags:
                            has_tag = True
                            break
                if has_tag:
                    sections.append(sec)
            if sections:
                return self.copy(sections=sections)
        return None

    def remove_tag(self, tag: Tag) -> Self:
        """Remove the specified tag from the message."""
        if tag in self.tags:
            self = self.copy(tags=self.tags - {tag})
        if self.tags_in_sections():
            sections = []
            for sec in self.sections:
                if isinstance(sec, Section) and tag in sec.tags:
                    sections.append(sec.copy(tags=sec.tags - {tag}))
                else:
                    sections.append(sec)
            if sections:
                self = self.copy(sections=sections)
        return self

    def add_tag(self, tag: Tag):
        self.tags.add(tag)

    def append(self, section: Section | str):
        # Ensure list[SectionType] by converting strings to Section
        if isinstance(section, str):
            section = Section(section)
        self.sections.append(section)

    def add_section(self, section: Section | str) -> Self:
        """Create a copy of the message with an additional section."""
        new_sections = self.sections.copy()
        if isinstance(section, str):
            new_sections.append(Section(section))
        else:
            new_sections.append(section)
        return self.copy(sections=new_sections)


class Message(BaseMessage):
    role: Role
    sections: list[SectionType]
    tags: set[Tag]

    # Unannotated attributes are not copied by copy()
    recipient = None
    target = False

    def __init__(
        self,
        role: Role,
        sections: str | Sequence[SectionType | str],
        *,
        tags: set[Tag] | None = None,
        eot_target: bool | None = None,
        strict: bool = True,  # Whether to check the roles
    ):
        if isinstance(sections, str):
            sections = [sections]
        for section in sections:
            if hasattr(section, "recipient"):
                assert section.recipient is None, "Sections with recipient cannot be used in Message"  # type: ignore

        super().__init__(role, sections, tags=tags, eot_target=eot_target, strict=strict)

    def to_serialized(self) -> dict[str, str | list[dict[str, Any]]]:
        return {
            "role": self.role,
            "content": [
                s.to_dict() if hasattr(s, "to_dict") else {"type": "text", "text": s}  # type: ignore
                for s in self.sections
            ],
            "tags": tags_to_str(self.tags),
        }

    @classmethod
    def from_serialized(cls, data: dict) -> Message:
        role = data["role"]
        sections = []
        for section in data["content"]:
            if section["type"] == "text":
                sections.append(
                    Section(section["text"], tags=tags_from_str(section.get("tags", "")))
                )
            elif section["type"] == "image":
                sections.append(Image(encoded=section["image"]))
            else:
                raise ValueError(f"Unknown section type: {section['type']}")

        tags = tags_from_str(data.get("tags", ""), strict=False)
        return cls(role, sections, tags=tags, strict=False)

    def to_st_message(self) -> STMessage:
        return STMessage(
            role=self.role,
            sections=self.sections,
            tags=self.tags,
        )


class STMessage(BaseMessage):
    role: Role
    sections: list[SectionType]
    tags: set[Tag]

    recipient: Recipient = None
    target: bool = False
    eot_target: bool | None = None

    def __init__(
        self,
        role: Role,
        sections: str | Sequence[SectionType | str],
        *,
        recipient: Recipient = None,
        tags: set[Tag] | None = None,
        target: bool = False,
        eot_target: bool | None = None,
    ):
        super().__init__(role, sections, tags=tags, target=target, eot_target=eot_target)

        if recipient is not None and self.recipient_in_sections():
            raise ValueError("recipient cannot be used with sections that already have a recipient")
        if recipient is not None and self.contains_target:
            raise ValueError("recipient and target cannot be used together")
        self.recipient = recipient

        # if self.contains_target and role != "assistant":
        #         raise ValueError("targets can be used only in assistant messages")

        if role in ("system", "user") and target:
            raise ValueError(f"When role is system or user, no target allowed ({target=})")

    @property
    def contains_recipient(self) -> bool:
        return self.recipient_in_sections() or self.recipient is not None

    def dropout_in_sections(self) -> bool:
        return any(getattr(section, "recipient", None) in {"student_dropout", "only_student_dropout"} for section in self.sections)

    def recipient_in_sections(self) -> bool:
        return any(getattr(section, "recipient", None) is not None for section in self.sections)

    def to_message(self, teacher: bool = True, dropout_rate: float | None = None) -> Message:
        if self.recipient and drop(self.recipient, teacher, dropout_rate):  # type: ignore
            return Message.empty()

        sections = [
            section.render(teacher, dropout_rate) if hasattr(section, "render") else section  # type: ignore
            for section in self.sections
        ]
        if self.target:
            for i, section in enumerate(sections):
                if hasattr(section, "target"):
                    section.target = True  # type: ignore
                else:
                    assert isinstance(section, str)
                    sections[i] = Section(section, target=True)

        return Message(self.role, sections, tags=self.tags)

    def apply_dropout(self, dropout_rate: float) -> STMessage:
        """Replace "student_dropout" sections with "teacher" or None, and
        "only_student_dropout" sections with "student" or drop them entirely."""
        if self.recipient == "student_dropout":
            recipient = None if random.random() > dropout_rate else "teacher"
            return self.copy(recipient=recipient)
        if self.recipient == "only_student_dropout":
            # Keep for student only with probability 1 - rate, otherwise drop the message
            if random.random() > dropout_rate:
                return self.copy(recipient="student")
            else:
                return STMessage.empty()

        new_sections = []
        for section in self.sections:
            try:
                if new_section := section.apply_dropout(dropout_rate):  # type: ignore
                    new_sections.append(new_section)
            except AttributeError:
                new_sections.append(section)

        return self.copy(sections=new_sections)

    def keep_recipients(self, recipients: list[Recipient]) -> STMessage:
        """Keep only sections with the given recipients. If the whole message has a recipient, keep it only if it's in the list."""
        if self.recipient and self.recipient not in recipients:
            return STMessage.empty()

        new_sections = [
            section
            for section in self.sections
            if isinstance(section, Section) and section.recipient in recipients
        ]

        return self.copy(sections=new_sections)


def merge_messages(messages: list[Message]) -> list[Message]:
    """Merge consecutive messages of the same role into one message.
    Combines sections and preserves tags from both messages.
    """
    merged_messages: list[Message] = []
    new_message: Message | None = None
    for message in messages:
        if new_message is None:
            new_message = message.copy()
        elif message.role == new_message.role:
            # Concatenate sections with a blank separator between merged messages
            combined_sections: list[SectionType] = list(new_message.sections)
            if combined_sections and message.sections:
                combined_sections.append(Section("\n\n"))
            combined_sections.extend(message.sections)
            # Merge tags at the message level
            combined_tags = set(new_message.tags) | set(message.tags)
            new_message = new_message.copy(sections=combined_sections, tags=combined_tags)
        else:
            merged_messages.append(new_message)
            new_message = message.copy()
    if new_message is not None:
        merged_messages.append(new_message)

    return merged_messages

def parse_tags(
        content: str,
        tags: list[str],
        strip: bool = True,  # Strip away the tags and leading/trailing whitespace.
        match_index: int = 0,  # The index of the match to return. 0 is the first match, -1 the last, etc.
        must_parse: bool = False,  # Raise an error if any tag is not found.
    ) -> dict[str, str]:
    """Parse the response from the LLM and extract the given tags."""
    parsed = {}
    for tag in tags:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match_all = re.finditer(pattern, content, re.DOTALL)
        if match_index == 0:
            try:
                match = next(match_all)
            except StopIteration:
                match = None
        else:
            matches = list(match_all)
            try:
                match = matches[match_index]
            except IndexError:
                match = None
        if match:
            value = match.group(1)
            parsed[tag] = value.strip() if strip else f"<{tag}>{value}</{tag}>"
        else:
            if must_parse:
                raise ValueError(f"Could not find tag <{tag}> in content: {content}.")
    return parsed


def parse_tag(
    content: str,
    tag: str,
    strip: bool = True,  # Strip away the tags and leading/trailing whitespace.
    match_index: int = 0,  # The index of the match to return. 0 is the first match, -1 the last, etc.
    must_parse: bool = False,  # Raise an error if the tag is not found.
) -> str | None:
    """Parse the response from the LLM and extract the given tag."""
    parsed = parse_tags(content, [tag], strip=strip, match_index=match_index, must_parse=must_parse)
    return parsed.get(tag, None)


def any_unfinished(content: str, tags: list[str]):
    """Check if the LLM response has unfinished tags."""
    for tag in tags:
        if f"<{tag}>" in content and f"</{tag}>" not in content:
            return True
    return False
