"""Module for prompt templates and formatting"""

from typing import Optional, Union


class Prompter:
    """Base class for prompt formatting"""

    def __init__(self, prompt_style: Optional[str] = None):
        self.prompt_style = prompt_style or "alpaca"
        self.template = self._load_template()

    def _load_template(self) -> dict:
        """Load the prompt template based on style"""
        # Default Alpaca template
        return {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
            "response_split": "### Response:",
        }

    def build_prompt(
        self,
        instruction: str,
        input: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        """Build a prompt from instruction, input, and output"""
        if input:
            prompt = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            prompt = self.template["prompt_no_input"].format(instruction=instruction)

        if output:
            prompt = f"{prompt}{output}"

        return prompt

    def get_response(self, output: str) -> str:
        """Format the response/output"""
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(prompt_style='{self.prompt_style}')"


class AlpacaPrompter(Prompter):
    """Alpaca-style prompt formatting"""

    def __init__(self):
        super().__init__(prompt_style="alpaca")


class ChatMLPrompter(Prompter):
    """ChatML-style prompt formatting"""

    def __init__(self):
        super().__init__(prompt_style="chatml")

    def _load_template(self) -> dict:
        """Load ChatML template"""
        return {
            "prompt_input": (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "prompt_no_input": (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "response_split": "<|im_start|>assistant",
        }

    def build_prompt(
        self,
        instruction: str,
        input: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        """Build a ChatML prompt"""
        if input:
            prompt = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            prompt = self.template["prompt_no_input"].format(instruction=instruction)

        if output:
            prompt = f"{prompt}{output}<|im_end|>"

        return prompt


class VicunaPrompter(Prompter):
    """Vicuna-style prompt formatting"""

    def __init__(self):
        super().__init__(prompt_style="vicuna")

    def _load_template(self) -> dict:
        """Load Vicuna template"""
        return {
            "prompt_input": (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
                "USER: {instruction}\n{input}\nASSISTANT: "
            ),
            "prompt_no_input": (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
                "USER: {instruction}\nASSISTANT: "
            ),
            "response_split": "ASSISTANT:",
        }


class CompletionPrompter(Prompter):
    """Simple completion-style prompting (no template)"""

    def __init__(self):
        super().__init__(prompt_style="completion")

    def build_prompt(
        self,
        instruction: str,
        input: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        """For completion, just return the text as-is"""
        text = instruction
        if input:
            text = f"{text} {input}"
        if output:
            text = f"{text} {output}"
        return text