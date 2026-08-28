from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
import os
import torch
from typing import Sequence

from core import MODEL_MAP, FAMILY_MAP
from core.message import Message, SectionType, Section, Image, Tag

class BaseTokenizer:
    model_family: str

    @abstractmethod
    def encode(
        self,
        text: str,
    ) -> list[int]:
        ...

    @abstractmethod
    def batch_encode(
        self,
        texts: list[str],
    ) -> list[list[int]]:
        ...

    @abstractmethod
    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        ...

    def tokenize_for_inference(
        self,
        messages: Sequence[Message] | None = None,
        continue_last_message: bool = False,
    ) -> list[int]:
        if self.model_family == "qwen-vl":
            raise NotImplementedError("Tokenization for vision models is not implemented yet.")
        if messages is None or len(messages) == 0:
            return []
        tok_messages = TokMessages.from_messages(
            messages,
            tokenizer=self,
            end_last_message=not continue_last_message,
            add_assistant_header=not continue_last_message,
        )
        return tok_messages.token_ids

    def tokenize_for_logprobs(
        self,
        messages: Sequence[Message] | None = None,
        end_last_message: bool = False,
    ) -> list[int]:
        if self.model_family == "qwen-vl":
            raise NotImplementedError("Tokenization for vision models is not implemented yet.")
        if messages is None or len(messages) == 0:
            return []
        tok_messages = TokMessages.from_messages(
            messages,
            tokenizer=self,
            end_last_message=end_last_message,
            add_assistant_header=False,
        )
        return tok_messages.token_ids

    def tokenize_for_training(self, messages: Sequence[Message]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize messages for training.

        Returns:
            token_ids: A tensor of token IDs.
            target_mask: A boolean tensor of target masks.
        """
        if len(messages) == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.bool)

        tok_messages = TokMessages.from_messages(
            messages,
            tokenizer=self,
            end_last_message=messages[-1].finished,
            add_assistant_header=False,
        )
        return torch.tensor(tok_messages.token_ids), tok_messages.target_mask

    def get_step_lengths(self, nested_message_list: Sequence[Sequence[Message]]) -> list[int]:
        assert len(nested_message_list) > 0
        tok_messages_list = [
            TokMessages.from_messages(
                messages,
                tokenizer=self,
                end_last_message=messages[-1].finished,
                add_assistant_header=False,
                tokenize=False
            )
            for messages in nested_message_list
        ]
        content_lists: list[list[list[str]]] = [tok_messages.get_content_list() for tok_messages in tok_messages_list]
        flat_content_list: list[str] = []
        step_num_sections: list[int] = []
        for content_list in content_lists:
            flattened_step_content = sum(content_list, [])
            step_num_sections.append(len(flattened_step_content))
            flat_content_list.extend(flattened_step_content)
        flat_tokenized_content_list: list[list[int]] = self.batch_encode(flat_content_list)
        tokenized_content_list: list[list[list[int]]] = [flat_tokenized_content_list[sum(step_num_sections[:i]):sum(step_num_sections[:i+1])] for i in range(len(step_num_sections))]
        step_lengths: list[int] = []
        for tokenized_content, tok_messages in zip(tokenized_content_list, tok_messages_list, strict=True):
            step_length = sum(len(tokens) for tokens in tokenized_content)
            if tok_messages.bos_id is not None:
                step_length += 1
            step_lengths.append(step_length)
        return step_lengths

class Tokenizer(BaseTokenizer):
    """A tokenizer that uses Hugging Face transformers library to encode and decode prompts."""
    def __init__(self, tokenizer_id: str):
        from transformers import AutoTokenizer

        assert isinstance(tokenizer_id, str), "Tokenizer must be a string"
        tokenizer_id = MODEL_MAP.get(tokenizer_id, tokenizer_id)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        self.model_family = FAMILY_MAP[tokenizer_id]

    def encode(self, text: str) -> list[int]:
        "IMPORTANT: We do not add special tokens!"
        if len(text) == 0:
            return []
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        "IMPORTANT: We do not add special tokens!"
        batch_tokens = self.tokenizer(
            texts,
            add_special_tokens=False,
        )['input_ids']
        return batch_tokens

    def decode(self, token_ids: list[int]) -> str:
        if len(token_ids) == 0:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


class DictTemplate:
    prefix: str
    suffix: str
    add_index: bool

    def __init__(self, prefix: str = "", suffix: str = "", add_index: bool = True):
        self.prefix = prefix
        self.suffix = suffix
        self.add_index = add_index

    def __getitem__(self, index: str):
        return self.prefix + index + self.suffix if self.add_index else self.prefix + self.suffix


SYSTEM_MESSAGES = {
    "llama": f"""Cutting Knowledge Date: December 2023
Today Date: {datetime.now().strftime("%d %b %Y")}

""",
    "qwen": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}

BOS = {
    "llama": "<|begin_of_text|>",
    "deepseek": "<｜begin▁of▁sentence｜>",
    "qwen": "",
    "qwen-vl": "",
}

BOS_IDS = {
    "llama": 128000,
    "qwen": None,
    "qwen-vl": None,
}

HEADERS = {
    "llama": {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "deepseek": {
        "user": "<｜User｜>",
        "assistant": "<｜Assistant｜>",
    },
    "qwen": DictTemplate(prefix="<|im_start|>", suffix="\n"),
    "qwen-vl": DictTemplate(prefix="<|im_start|>", suffix="\n"),
}

FOOTERS = {
    "llama": {
        "system": "<|eot_id|>",
        "user": "<|eot_id|>",
        "assistant": "<|eot_id|>",
    },
    "deepseek": {
        "user": "",
        "assistant": "<｜end▁of▁sentence｜>",
    },
    "qwen": DictTemplate(prefix="<|im_end|>\n", add_index=False),
    "qwen-vl": DictTemplate(prefix="<|im_end|>\n", add_index=False),
}

IMAGE_PLACEHOLDER = {
    "qwen-vl": "<|vision_start|><|image_pad|><|vision_end|>"
}


class TokSection:
    prompt: str
    _token_ids: list[int] | None
    target: bool = False
    tags: set[Tag]

    def __init__(self, prompt: str, token_ids: list[int] | None = None, target: bool = False, tags: set | None = None):
        self.prompt = prompt
        self._token_ids = token_ids
        self.target = target
        self.tags = tags or set()

    def __repr__(self):
        s = f"{self.__class__.__name__}({self.prompt!r}"
        s += f", token_ids={self._token_ids}"
        if self.target:
            s += f", target={self.target}"
        if self.tags:
            s += f", tags={self.tags}"
        s += ")"
        return s

    @property
    def token_ids(self) -> list[int]:
        if not self.is_tokenized:
            raise ValueError("Trying to access token_ids of a section that has not been tokenized")
        return self._token_ids  # type: ignore

    @property
    def is_tokenized(self) -> bool:
        return self._token_ids is not None

    def tokenize(self, tokenizer: BaseTokenizer) -> None:
        self._token_ids = tokenizer.encode(self.prompt)

    @property
    def target_mask(self) -> torch.Tensor:
        if self._token_ids is None:
            raise ValueError("Trying to access target_mask of a message that has not been tokenized")
        if self.target:
            return torch.ones(len(self._token_ids), dtype=torch.bool)
        else:
            return torch.zeros(len(self._token_ids), dtype=torch.bool)

    @classmethod
    def from_section(
        cls,
        section: SectionType,
        tokenizer: BaseTokenizer,
    ) -> TokSection:
        model_family = tokenizer.model_family
        if isinstance(section, str):
            prompt = section
        elif isinstance(section, Section) and isinstance(section.content, str):
            prompt = section.content
        elif isinstance(section, Image) or (
            isinstance(section, Section) and isinstance(section.content, Image)
        ):
            try:
                prompt = IMAGE_PLACEHOLDER[model_family]
            except KeyError as e:
                raise ValueError(f"Model family '{model_family}' is not supported for image sections") from e
        else:
            raise ValueError(f"Unknown section type: {type(section)}")

        tags = set(Tag(tag) for tag in getattr(section, "tags", {}))

        return cls(
            prompt=prompt,
            token_ids=None,
            target=getattr(section, "target", False),
            tags=tags,
        )


class TokMessage:
    """A message that is used to generate a prompt."""
    header: TokSection
    sections: list[TokSection]
    footer: TokSection | None = None
    tags: set[Tag]

    def __init__(
        self,
        header: TokSection,
        sections: list[TokSection],
        footer: TokSection | None = None,
        tags: set | None = None,
    ):
        self.header = header
        self.sections = sections
        self.footer = footer
        self.tags = tags or set()

    @property
    def token_ids(self) -> list[int]:
        assert self.is_tokenized, "Trying to access token_ids of a message that has not been tokenized"
        token_ids: list[int] = self.header._token_ids  # type: ignore
        token_ids = sum([section._token_ids for section in self.sections], token_ids)  # type: ignore
        if self.footer:
            token_ids += self.footer._token_ids  # type: ignore
        return token_ids

    @property
    def is_tokenized(self) -> bool:
        if not self.header.is_tokenized:
            return False
        if any(not section.is_tokenized for section in self.sections):
            return False
        if self.footer and not self.footer.is_tokenized:
            return False
        return True

    def get_content_list(self) -> list[str]:
        content_list = [self.header.prompt]
        content_list.extend([section.prompt for section in self.sections])
        if self.footer:
            content_list.append(self.footer.prompt)
        return content_list

    def tokenize(self, tokenizer: BaseTokenizer) -> None:
        self.header.tokenize(tokenizer)
        for section in self.sections:
            section.tokenize(tokenizer)
        if self.footer:
            self.footer.tokenize(tokenizer)

    def _distribute_token_ids(self, token_ids: list[list[int]]) -> None:
        self.header._token_ids = token_ids[0]
        token_ids = token_ids[1:]
        for section, section_token_ids in zip(self.sections, token_ids):
            section._token_ids = section_token_ids
        if self.footer:
            assert len(token_ids) == len(self.sections) + 1
            self.footer._token_ids = token_ids[-1]

    @property
    def target_mask(self) -> torch.Tensor:
        masks: list[torch.Tensor] = [self.header.target_mask]
        masks.extend([section.target_mask for section in self.sections])
        if self.footer:
            masks.append(self.footer.target_mask)
        return torch.cat(masks, dim=0)

    def get_content_indices(self, tag: Tag | None = None, offset: int = 0) -> tuple[list[int], int]:
        """Get indices of sections and footer tokens. Optionally, return only indices
        of messages and sections with the given tag.
        """
        assert self.is_tokenized, "Trying to access content indices of a message that has not been tokenized"

        # We always skip the header
        offset += len(self.header._token_ids)  # type: ignore

        indices = []
        msg_accept = tag is None or tag in self.tags
        for section in self.sections:
            section_len = len(section._token_ids)  # type: ignore
            if msg_accept or tag in section.tags:
                indices.extend(
                    [i + offset for i in range(section_len)]
                )
            offset += section_len
        footer_len = 0 if self.footer is None else len(self.footer._token_ids)  # type: ignore
        if footer_len and (
            msg_accept
            or (self.sections and tag in self.sections[-1].tags)
        ):
            # The footer is included if the messaged has the tag or the last section has the tag
            indices.extend(
                [i + offset for i in range(footer_len)]
            )

        # Even if the footer is not included, we still need to update the offset
        offset += footer_len
        return indices, offset

    @property
    def prompt(self):
        """Return the prompt of the message."""
        prompt = self.header.prompt + "".join(section.prompt for section in self.sections)
        if self.footer:
            prompt += self.footer.prompt
        return prompt

    @classmethod
    def from_message(
        cls,
        message: Message,
        tokenizer: BaseTokenizer,
        add_footer: bool = True,
    ) -> TokMessage | None:
        """Constructor used for training."""
        if not message:
            return None

        model_family = tokenizer.model_family
        role = message.role
        header = TokSection.from_section(HEADERS[model_family][role], tokenizer)
        sections = [
            TokSection.from_section(section, tokenizer)
            for section in message.sections if section
        ]
        if add_footer:
            footer = TokSection.from_section(FOOTERS[model_family][role], tokenizer)
            footer.target = sections[-1].target if message.eot_target is None else message.eot_target
        else:
            footer = None

        tags = set(Tag(tag) for tag in message.tags)

        return cls(header, sections, footer, tags=tags)

    @classmethod
    def only_header(cls, role: str, tokenizer: BaseTokenizer) -> TokMessage:
        return cls.from_message(  # type: ignore
            Message(role=role, sections=[]),
            tokenizer,
            add_footer=False
        )

    def __repr__(self):
        s = f"{self.__class__.__name__}(header={self.header}"
        if self.sections:
            s += f",\nsections=[{', '.join(repr(s) for s in self.sections)}]"
        if self.footer:
            s += f", footer={self.footer}"
        if self.tags:
            s += f", tags={self.tags}"
        s += ")"
        return s


class TokMessages(list[TokMessage]):
    """A list of messages that is used to generate a prompt."""
    bos_id: int | None
    model_family: str

    def __init__(self, messages: list[TokMessage], tokenizer: BaseTokenizer):
        super().__init__(messages)
        self.model_family = tokenizer.model_family
        self.bos_id = BOS_IDS.get(self.model_family, None)

    def get_prompt(self, add_bos: bool = True) -> str:
        prompt = BOS[self.model_family] if add_bos else ""
        prompt += "".join(msg.prompt for msg in self)
        return prompt

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Message],
        tokenizer: BaseTokenizer,
        end_last_message: bool = True,
        add_assistant_header: bool = True,
        tokenize: bool = True,
    ):
        tok_messages = [
            TokMessage.from_message(
                msg,
                tokenizer,
                add_footer=msg is not messages[-1] or end_last_message,
            )
            for msg in messages if msg
        ]

        if add_assistant_header:
            tok_messages.append(TokMessage.only_header("assistant", tokenizer))

        tok_messages = cls(tok_messages, tokenizer)  # type: ignore
        if tokenize:
            tok_messages.tokenize(tokenizer)
        return tok_messages

    @property
    def token_ids(self) -> list[int]:
        token_ids = [self.bos_id] if self.bos_id is not None else []
        token_ids = sum([msg.token_ids for msg in self], token_ids)
        return token_ids

    def tokenize(self, tokenizer: BaseTokenizer) -> None:
        content_list: list[list[str]] = self.get_content_list()
        flattened_content_list: list[str] = sum(content_list, [])
        flattened_messages_token_ids: list[list[int]] = tokenizer.batch_encode(flattened_content_list)
        for msg, msg_content in zip(self, content_list, strict=True):
            n_sections = len(msg_content)
            msg_token_ids = flattened_messages_token_ids[:n_sections]
            flattened_messages_token_ids = flattened_messages_token_ids[n_sections:]
            msg._distribute_token_ids(msg_token_ids)

    def get_content_list(self) -> list[list[str]]:
        return [msg.get_content_list() for msg in self]

    def get_content_indices(self, tag: Tag | None = None) -> list[int]:
        """Get indices of sections and footer tokens. Optionally, return only indices
        of messages and sections with the given tag.
        """
        offset = 0 if self.bos_id is None else 1
        indices = []
        for msg in self:
            msg_indices, offset = msg.get_content_indices(tag, offset)
            indices.extend(msg_indices)

        return indices

    @property
    def target_mask(self) -> torch.Tensor:
        masks = [torch.tensor([False])] if self.bos_id is not None else []
        masks.extend([msg.target_mask for msg in self])
        if len(masks) == 0:
            return torch.tensor([], dtype=torch.bool)
        return torch.cat(masks, dim=0)

    def __repr__(self):
        return f"{self.__class__.__name__}(messages=[{', '.join(repr(m) for m in self)}], model_family={self.model_family})"


def apply_chat_template(
    tokenizer: BaseTokenizer,  # TODO: Only use model_family, not the whole tokenizer
    ser_messages: list[dict],
    end_last_message: bool = True,
    add_assistant_header: bool = True,
) -> str:
    """This function is deprecated. Use Tokenizer.tokenize_for_inference instead.

    This function can be used to convert messages to a prompt for further tokenization during training.

    It can also be used to compare the prompt with the result of apply_chat_template provided
    by HF tokenizers.
    """
    messages = [Message.from_serialized(msg) for msg in ser_messages]
    tok_messages = TokMessages.from_messages(messages, tokenizer, end_last_message, add_assistant_header)
    return tok_messages.get_prompt()


def add_new_line_symbols(text: str) -> str:
    """Replace new line characters with the NEW_LINE symbol."""
    new_line = "\u00B6"
    return text.replace("\n", f"{new_line}\n")


def print_tokenized(text: str, tokenizer: BaseTokenizer):
    colors = [150, 208, 190, 188, 226, 156, 153, 230, 195, 213, 194, 173]
    token_ids = tokenizer.encode(text)
    for i, token_id in enumerate(token_ids):
        decoded = add_new_line_symbols(tokenizer.decode([token_id]))
        # Print with a random background color
        c = colors[i % len(colors)]
        print(f"\033[48;5;{c}m{decoded}", end="")
    print("\033[0m")
