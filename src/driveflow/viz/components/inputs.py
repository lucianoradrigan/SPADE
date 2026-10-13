"""Shared input widgets. slider_with_custom used to be copy-pasted into three dashboard modules
(each module was forbidden from importing dashboard.py); it now lives here once."""


def slider_with_custom(container, label, min_value, max_value, value, step=None, format=None, help=None, key=None):
    """A slider for the common range, plus a "Custom value" checkbox that swaps in an unbounded
    number_input -- a slider alone can never go past max_value, and some parameters (stress-test
    severities, below-floor load resistance) legitimately need to. Keeps int vs float from
    `value` (downstream np.linspace num=/range-like loops need a real int)."""
    base_key = key or label
    slider_val = container.slider(label, min_value, max_value, value, step=step, format=format, help=help, key=f"{base_key}_slider")
    use_custom = container.checkbox("Custom value", key=f"{base_key}_custom", help=f'Type a "{label}" value outside {min_value}-{max_value} above.')
    if not use_custom:
        return slider_val
    is_int = isinstance(value, int) and not isinstance(value, bool)
    custom_val = container.number_input(
        f"{label} (custom)",
        value=int(slider_val) if is_int else float(slider_val),
        step=step or (1 if is_int else 0.1),
        format=format,
        key=f"{base_key}_custom_input",
    )
    return int(custom_val) if is_int else custom_val
