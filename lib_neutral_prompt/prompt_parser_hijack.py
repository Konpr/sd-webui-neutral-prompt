from typing import List
from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import script_callbacks, prompt_parser

# Устанавливаем hijacker на новую функцию
prompt_parser_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=prompt_parser,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


@prompt_parser_hijacker.hijack('get_learned_conditioning_prompt_schedules')
def get_learned_conditioning_prompt_schedules_hijack(prompts: List[str], base_steps: int, *args, original_function=None, **kwargs):
    """
    Хукаем планировщик кондишенов, чтобы встроить поддержку neutral prompt.
    """
    # Получаем расписания через оригинальную функцию WebUI
    schedules = original_function(prompts, base_steps, *args, **kwargs)

    # Парсим prompt'ы через парсер neutral_prompt
    try:
        global_state.prompt_exprs = parse_prompts(prompts)
    except Exception as e:
        print("[neutral_prompt] Failed to parse prompts:", e)
        global_state.prompt_exprs = []

    return schedules


def parse_prompts(prompts: List[str]) -> List[neutral_prompt_parser.PromptExpr]:
    exprs = []
    for prompt in prompts:
        expr = neutral_prompt_parser.parse_root(prompt)
        exprs.append(expr)
    return exprs


def transpile_exprs(exprs: List[neutral_prompt_parser.PromptExpr]) -> List[str]:
    return [expr.accept(WebuiPromptVisitor()) for expr in exprs]


class WebuiPromptVisitor:
    def visit_leaf_prompt(self, that: neutral_prompt_parser.LeafPrompt) -> str:
        return f'{that.prompt} :{that.weight}'

    def visit_composite_prompt(self, that: neutral_prompt_parser.CompositePrompt) -> str:
        return ' AND '.join(child.accept(self) for child in that.children)
