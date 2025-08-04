from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import script_callbacks, sd_samplers, shared
from typing import List, Tuple
import functools
import torch
import dataclasses
import textwrap
import sys


@dataclasses.dataclass
class CombineDenoiseArgs:
    x_out: torch.Tensor
    uncond: torch.Tensor
    cond_indices: List[Tuple[int, float]]


def combine_denoised_hijack(
    x_out: torch.Tensor,
    batch_cond_indices: List[List[Tuple[int, float]]],
    text_uncond: torch.Tensor,
    cond_scale: float,
    original_function,
) -> torch.Tensor:
    if not global_state.is_enabled or not global_state.prompt_exprs:
        return original_function(x_out, batch_cond_indices, text_uncond, cond_scale)

    denoised = get_webui_denoised(x_out, batch_cond_indices, text_uncond, cond_scale, original_function)
    uncond = x_out[-text_uncond.shape[0]:]

    for batch_i, (prompt_expr, cond_indices) in enumerate(zip(global_state.prompt_exprs, batch_cond_indices)):
        args = CombineDenoiseArgs(x_out, uncond[batch_i], cond_indices)

        try:
            cond_delta = prompt_expr.accept(CondDeltaVisitor(), args, 0)
            aux_delta = prompt_expr.accept(AuxCondDeltaVisitor(), args, cond_delta, 0)
            cfg_cond = denoised[batch_i] + aux_delta * cond_scale
            denoised[batch_i] = cfg_rescale(cfg_cond, uncond[batch_i] + cond_delta + aux_delta)
        except Exception as e:
            print(f"[neutral_prompt] Error in denoising pass for sample {batch_i}: {e}", file=sys.stderr)

    return denoised


def get_webui_denoised(x_out, batch_cond_indices, text_uncond, cond_scale, original_function):
    uncond = x_out[-text_uncond.shape[0]:]
    sliced_x_out = []
    sliced_batch_cond_indices = []

    for batch_i, (prompt, cond_indices) in enumerate(zip(global_state.prompt_exprs, batch_cond_indices)):
        args = CombineDenoiseArgs(x_out, uncond[batch_i], cond_indices)
        sliced, indices = gather_webui_conds(prompt, args, 0, len(sliced_x_out))
        if indices:
            sliced_batch_cond_indices.append(indices)
        sliced_x_out.extend(sliced)

    sliced_x_out += list(uncond)
    return original_function(torch.stack(sliced_x_out, dim=0), sliced_batch_cond_indices, text_uncond, cond_scale)


def gather_webui_conds(prompt, args: CombineDenoiseArgs, index_in: int, index_out: int):
    sliced, indices = [], []
    for child in prompt.children:
        if child.conciliation is None:
            if isinstance(child, neutral_prompt_parser.LeafPrompt):
                child_x_out = args.x_out[args.cond_indices[index_in][0]]
            else:
                child_x_out = child.accept(CondDeltaVisitor(), args, index_in)
                child_x_out += child.accept(AuxCondDeltaVisitor(), args, child_x_out, index_in)
                child_x_out += args.uncond

            sliced.append(child_x_out)
            indices.append((index_out + len(sliced) - 1, child.weight))

        index_in += child.accept(neutral_prompt_parser.FlatSizeVisitor())

    return sliced, indices


def cfg_rescale(cfg_cond, cond):
    if global_state.cfg_rescale == 0:
        return cfg_cond

    global_state.apply_and_clear_cfg_rescale_override()
    mean = (1 - global_state.cfg_rescale) * cfg_cond.mean() + global_state.cfg_rescale * cond.mean()
    factor = global_state.cfg_rescale * (cond.std() / cfg_cond.std() - 1) + 1
    return mean + (cfg_cond - cfg_cond.mean()) * factor


class CondDeltaVisitor:
    def visit_leaf_prompt(self, that, args: CombineDenoiseArgs, index: int):
        ref = args.cond_indices[index]
        return args.x_out[ref[0]] - args.uncond

    def visit_composite_prompt(self, that, args: CombineDenoiseArgs, index: int):
        result = torch.zeros_like(args.x_out[0])
        for child in that.children:
            if child.conciliation is None:
                child_delta = child.accept(CondDeltaVisitor(), args, index)
                child_delta += child.accept(AuxCondDeltaVisitor(), args, child_delta, index)
                result += child.weight * child_delta
            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())
        return result


class AuxCondDeltaVisitor:
    def visit_leaf_prompt(self, that, args: CombineDenoiseArgs, delta, index: int):
        return torch.zeros_like(args.x_out[0])

    def visit_composite_prompt(self, that, args: CombineDenoiseArgs, delta, index: int):
        result = torch.zeros_like(args.x_out[0])
        salient = []

        for child in that.children:
            if child.conciliation is not None:
                child_delta = child.accept(CondDeltaVisitor(), args, index)
                child_delta += child.accept(AuxCondDeltaVisitor(), args, child_delta, index)

                if child.conciliation.name == "PERPENDICULAR":
                    result += child.weight * get_perpendicular_component(delta, child_delta)
                elif child.conciliation.name == "SALIENCE_MASK":
                    salient.append((child_delta, child.weight))
                elif child.conciliation.name == "SEMANTIC_GUIDANCE":
                    result += child.weight * filter_abs_top_k(child_delta, 0.05)

            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        result += salient_blend(delta, salient)
        return result


def get_perpendicular_component(normal, vector):
    if (normal == 0).all():
        if shared.state.sampling_step <= 0:
            warn_projection_not_found()
        return vector
    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


def salient_blend(normal, vectors):
    salience = [get_salience(normal)] + [get_salience(v[0]) for v in vectors]
    mask = torch.argmax(torch.stack(salience, dim=0), dim=0)

    result = torch.zeros_like(normal)
    for i, (vector, weight) in enumerate(vectors, start=1):
        vector_mask = (mask == i).float()
        result += weight * vector_mask * (vector - normal)
    return result


def get_salience(vector):
    return torch.softmax(torch.abs(vector).flatten(), dim=0).reshape_as(vector)


def filter_abs_top_k(vector, k_ratio):
    k = int(torch.numel(vector) * (1 - k_ratio))
    top_k, _ = torch.kthvalue(torch.abs(torch.flatten(vector)), k)
    return vector * (torch.abs(vector) >= top_k).to(vector.dtype)


# Устанавливаем hijack на create_sampler
sd_samplers_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=sd_samplers,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


@sd_samplers_hijacker.hijack('create_sampler')
def create_sampler_hijack(name: str, model, original_function):
    sampler = original_function(name, model)
    if not hasattr(sampler, 'model_wrap_cfg') or not hasattr(sampler.model_wrap_cfg, 'combine_denoised'):
        if global_state.is_enabled:
            warn_unsupported_sampler()
        return sampler

    sampler.model_wrap_cfg.combine_denoised = functools.partial(
        combine_denoised_hijack,
        original_function=sampler.model_wrap_cfg.combine_denoised,
    )
    return sampler


def warn_unsupported_sampler():
    console_warn("""
        Neutral prompt не может быть применён к DDIM / PLMS / UniPC
        Расширение не будет использоваться с этим семплером.
    """)


def warn_projection_not_found():
    console_warn("""
        Невозможно найти проекцию для AND_PERP — результат может быть неточным.
    """)


def console_warn(message: str):
    if not global_state.verbose:
        return
    print(f"\n[neutral_prompt]: {textwrap.dedent(message)}", file=sys.stderr)
