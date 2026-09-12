"""Terminal selectors, with a line-input fallback for scripts."""

from pathlib import Path
from contextlib import closing
from typing import TextIO
from collections.abc import Callable


class Cancelled(Exception):
    pass


def interactive(stdin: TextIO, stdout: TextIO) -> bool:
    return bool(getattr(stdin, "isatty", lambda: False)() and getattr(stdout, "isatty", lambda: False)())


def select(message: str, choices: list[str], *, default: int = 0, stdin: TextIO, stdout: TextIO) -> int:
    if interactive(stdin, stdout):
        import questionary
        from prompt_toolkit.input import create_input
        from prompt_toolkit.output import create_output
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.margins import ScrollbarMargin
        from questionary.prompts.common import InquirerControl

        items = [questionary.Choice(label, value=index) for index, label in enumerate(choices)]
        # questionary uses `default` for both the initial cursor and a persistent
        # "selected" style. We only have a single-choice menu: keeping that
        # value makes the first row look selected forever after the cursor moves.
        initial = min(max(default, 0), len(items) - 1)
        with closing(create_input(stdin=stdin)) as terminal_input:
            try:
                question = questionary.select(
                    message,
                    choices=items,
                    default=None,
                    qmark="◆",
                    pointer="❯",
                    style=questionary.Style([
                        ("qmark", "fg:#5fd7d7 bold"),
                        ("question", "bold"),
                        ("pointer", "fg:#5fd7d7 bold"),
                        ("highlighted", "fg:#5fd7d7 bold"),
                        ("answer", "fg:#5fd7d7"),
                        ("instruction", "fg:#808080"),
                    ]),
                    use_shortcuts=False,
                    use_jk_keys=False,
                    instruction="\n  ↑/↓ Move   Enter Select   Esc Back\n",
                    input=terminal_input,
                    output=create_output(stdout=stdout),
                )
                question.application.ttimeoutlen = 0.1
                question.application.timeoutlen = 0.1
                for window in question.application.layout.find_all_windows():
                    if isinstance(window.content, InquirerControl):
                        window.content.default = None
                        window.content.pointed_at = initial
                        window.height = Dimension(max=8)
                        if len(choices) > 8:
                            window.right_margins = [ScrollbarMargin(display_arrows=True)]
                @question.application.key_bindings.add("escape")
                def cancel(event):
                    event.app.erase_when_done = True
                    event.app.exit(exception=KeyboardInterrupt())
                return question.unsafe_ask()
            except (KeyboardInterrupt, EOFError) as error:
                raise Cancelled from error

    for index, choice in enumerate(choices, 1):
        stdout.write(f"  {index}) {choice}\n")
    stdout.write(f"{message} [{default + 1}]: ")
    stdout.flush()
    line = stdin.readline()
    if not line:
        raise Cancelled
    answer = line.strip() or str(default + 1)
    if not answer.isdigit() or not 1 <= int(answer) <= len(choices):
        raise ValueError(f"Not a listed choice: {answer}")
    return int(answer) - 1


def text(message: str, *, stdin: TextIO, stdout: TextIO, default: str = "", secret: bool = False, validate: Callable[[str], bool | str] | None = None) -> str:
    if interactive(stdin, stdout):
        import questionary
        from prompt_toolkit.input import create_input
        from prompt_toolkit.output import create_output
        from prompt_toolkit.key_binding import KeyBindings

        with closing(create_input(stdin=stdin)) as terminal_input:
            try:
                prompt = questionary.password if secret else questionary.text
                keys = KeyBindings()
                @keys.add("escape")
                def cancel(event):
                    event.app.erase_when_done = True
                    event.app.exit(exception=KeyboardInterrupt())
                question = prompt(message, default=default, validate=validate, qmark="◆", instruction="(Esc to go back)", key_bindings=keys, input=terminal_input, output=create_output(stdout=stdout))
                question.application.ttimeoutlen = 0.1
                question.application.timeoutlen = 0.1
                return question.unsafe_ask()
            except (KeyboardInterrupt, EOFError) as error:
                raise Cancelled from error
    suffix = f" [{default}]" if default else ""
    stdout.write(f"{message}{suffix}: ")
    stdout.flush()
    line = stdin.readline()
    if not line:
        raise Cancelled
    answer = line.strip() or default
    if validate:
        valid = validate(answer)
        if valid is not True:
            raise ValueError(valid or "Invalid answer.")
    return answer


def preset(message: str, labels: list[str], values: list[str], *, stdin: TextIO, stdout: TextIO, default: str = "", validate: Callable[[str], bool | str] | None = None) -> str:
    """Choose a preset; explicit custom entry is always last."""
    if not interactive(stdin, stdout):
        return text(message, default=default, validate=validate, stdin=stdin, stdout=stdout)
    initial = values.index(default) if default in values else len(values) if default else 0
    while True:
        selected = select(message, [*labels, "Type your own…"], default=initial, stdin=stdin, stdout=stdout)
        if selected < len(values):
            return values[selected]
        try:
            return text(message, default=default, validate=validate, stdin=stdin, stdout=stdout)
        except Cancelled:
            initial = selected


def path(*, stdin: TextIO, stdout: TextIO, start: str = "") -> str:
    if not interactive(stdin, stdout):
        return text("File or folder to search", default=".", stdin=stdin, stdout=stdout)
    initial = Path(start).expanduser().resolve() if start else Path.cwd()
    current = initial if initial.is_dir() else initial.parent
    highlighted = initial if initial.is_file() else None
    history = []
    while True:
        try:
            entries = sorted((entry for entry in current.iterdir() if not entry.name.startswith(".")), key=lambda entry: (not entry.is_dir(), entry.name.casefold()))
        except OSError as error:
            stdout.write(f"Cannot open {current}: {error}\n")
            entries = []
        options = ["Search this folder", "↑ Parent folder"]
        options.extend(f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries)
        options.extend(["Back to main menu", "Type a path…"])
        default = entries.index(highlighted) + 2 if highlighted in entries else 0
        try:
            choice = select(f"Which file or folder?\n  {current}", options, default=default, stdin=stdin, stdout=stdout)
        except Cancelled:
            if not history:
                raise
            current, highlighted = history.pop()
            continue
        if choice == 0:
            return str(current)
        if choice == 1:
            highlighted = current
            current = current.parent
            if history and history[-1][0] == current:
                history.pop()
        elif choice == len(options) - 1:
            try:
                value = text("Full path to a file or folder", default=str(current), stdin=stdin, stdout=stdout,
                             validate=lambda answer: True if Path(answer).expanduser().exists() else "This path does not exist. Edit it or press Esc to return.")
            except Cancelled:
                continue
            target = Path(value).expanduser()
            if target.exists():
                return str(target.resolve())
            stdout.write(f"Path not found: {value}\n")
        elif choice == len(options) - 2:
            raise Cancelled
        else:
            selected = entries[choice - 2]
            if selected.is_dir():
                history.append((current, selected))
                current = selected
                highlighted = None
            else:
                return str(selected)


def result_count(default: str, *, stdin: TextIO, stdout: TextIO) -> str:
    values = sorted(set(["1", "3", "5", "10", "20", default]), key=int)
    return preset("How many results would you like?", values, values, default=default, stdin=stdin, stdout=stdout,
                  validate=lambda answer: True if answer.strip().isdecimal() and int(answer.strip()) > 0 else "Enter a whole number greater than zero, or press Esc to choose from the list.").strip()
