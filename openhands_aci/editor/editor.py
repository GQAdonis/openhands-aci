import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Literal, get_args

from openhands_aci.linter import DefaultLinter
from openhands_aci.utils.shell import run_shell_cmd

from .config import SNIPPET_CONTEXT_WINDOW
from .exceptions import (
    EditorToolParameterInvalidError,
    EditorToolParameterMissingError,
    ToolError,
)
from .prompts import DIRECTORY_CONTENT_TRUNCATED_NOTICE, FILE_CONTENT_TRUNCATED_NOTICE
from .results import CLIResult, maybe_truncate

Command = Literal[
    'view',
    'create',
    'str_replace',
    'insert',
    'undo_edit',
    # 'jump_to_definition', TODO:
    # 'find_references' TODO:
]


class OHEditor:
    """
    An filesystem editor tool that allows the agent to
    - view
    - create
    - navigate
    - edit files
    The tool parameters are defined by Anthropic and are not editable.

    Original implementation: https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/computer_use_demo/tools/edit.py
    """

    TOOL_NAME = 'oh_editor'

    def __init__(self):
        self._file_history: dict[Path, list[str]] = defaultdict(list)
        self._linter = DefaultLinter()

    def __call__(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        enable_linting: bool = False,
        **kwargs,
    ) -> CLIResult:
        _path = Path(path)
        self.validate_path(command, _path)
        if command == 'view':
            return self.view(_path, view_range)
        elif command == 'create':
            if file_text is None:
                raise EditorToolParameterMissingError(command, 'file_text')
            self.write_file(_path, file_text)
            self._file_history[_path].append(file_text)
            return CLIResult(
                path=str(_path),
                new_content=file_text,
                prev_exist=False,
                output=f'File created successfully at: {_path}',
            )
        elif command == 'str_replace':
            if old_str is None:
                raise EditorToolParameterMissingError(command, 'old_str')
            if new_str == old_str:
                raise EditorToolParameterInvalidError(
                    'new_str',
                    new_str,
                    'No replacement was performed. `new_str` and `old_str` must be different.',
                )
            return self.str_replace(_path, old_str, new_str, enable_linting)
        elif command == 'insert':
            if insert_line is None:
                raise EditorToolParameterMissingError(command, 'insert_line')
            if new_str is None:
                raise EditorToolParameterMissingError(command, 'new_str')
            return self.insert(_path, insert_line, new_str, enable_linting)
        elif command == 'undo_edit':
            return self.undo_edit(_path)

        raise ToolError(
            f'Unrecognized command {command}. The allowed commands for the {self.TOOL_NAME} tool are: {", ".join(get_args(Command))}'
        )

    def str_replace(
        self, path: Path, old_str: str, new_str: str | None, enable_linting: bool
    ) -> CLIResult:
        """
        Implement the str_replace command, which replaces old_str with new_str in the file content.
        For large files, it uses a streaming approach to find occurrences and perform replacements.
        """
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ''

        # For small files, use the simpler approach
        if path.stat().st_size < 1024 * 1024:  # 1MB
            file_content = self.read_file(path).expandtabs()
            occurrences = file_content.count(old_str)
            if occurrences == 0:
                raise ToolError(
                    f'No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}.'
                )
            if occurrences > 1:
                # Find starting line numbers for each occurrence
                line_numbers = []
                start_idx = 0
                while True:
                    idx = file_content.find(old_str, start_idx)
                    if idx == -1:
                        break
                    line_num = file_content.count('\n', 0, idx) + 1
                    line_numbers.append(line_num)
                    start_idx = idx + 1
                raise ToolError(
                    f'No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {line_numbers}. Please ensure it is unique.'
                )

            new_file_content = file_content.replace(old_str, new_str)
            self.write_file(path, new_file_content)
            self._file_history[path].append(file_content)

            # Create a snippet of the edited section
            replacement_line = file_content.split(old_str)[0].count('\n')
            start_line = max(0, replacement_line - SNIPPET_CONTEXT_WINDOW)
            end_line = replacement_line + SNIPPET_CONTEXT_WINDOW + new_str.count('\n')
            snippet = '\n'.join(new_file_content.split('\n')[start_line : end_line + 1])

            success_message = f'The file {path} has been edited. '
            success_message += self._make_output(
                snippet, f'a snippet of {path}', start_line + 1
            )

            if enable_linting:
                lint_results = self._run_linting(file_content, new_file_content, path)
                success_message += '\n' + lint_results + '\n'

            success_message += 'Review the changes and make sure they are as expected. Edit the file again if necessary.'
            return CLIResult(
                output=success_message,
                prev_exist=True,
                path=str(path),
                old_content=file_content,
                new_content=new_file_content,
            )

        # For large files, use a streaming approach to count occurrences
        occurrences = []
        current_chunk = []
        chunk_start_line = 1
        line_number = 1
        
        # Read the file in chunks of lines to handle matches that span multiple lines
        with path.open() as f:
            for line in f:
                current_chunk.append(line)
                if len(current_chunk) >= 1000:  # Process 1000 lines at a time
                    chunk_text = ''.join(current_chunk).expandtabs()
                    if old_str in chunk_text:
                        # Count occurrences in this chunk
                        start = 0
                        while True:
                            pos = chunk_text.find(old_str, start)
                            if pos == -1:
                                break
                            # Calculate the line number for this occurrence
                            line_in_chunk = chunk_text.count('\n', 0, pos) + 1
                            occurrences.append(chunk_start_line + line_in_chunk - 1)
                            start = pos + 1
                    
                    # Keep last few lines in case match spans across chunks
                    overlap_lines = min(len(current_chunk), len(old_str.splitlines()) + 1)
                    current_chunk = current_chunk[-overlap_lines:]
                    chunk_start_line = line_number - len(current_chunk) + 1
                line_number += 1
            
            # Process the last chunk
            if current_chunk:
                chunk_text = ''.join(current_chunk).expandtabs()
                if old_str in chunk_text:
                    start = 0
                    while True:
                        pos = chunk_text.find(old_str, start)
                        if pos == -1:
                            break
                        line_in_chunk = chunk_text.count('\n', 0, pos) + 1
                        occurrences.append(chunk_start_line + line_in_chunk - 1)
                        start = pos + 1

        if not occurrences:
            raise ToolError(
                f'No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}.'
            )
        if len(occurrences) > 1:
            raise ToolError(
                f'No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {occurrences}. Please ensure it is unique.'
            )

        # We found exactly one occurrence, now read the content around it
        occurrence_line = occurrences[0]
        start_line = max(1, occurrence_line - SNIPPET_CONTEXT_WINDOW)
        end_line = occurrence_line + SNIPPET_CONTEXT_WINDOW + new_str.count('\n')
        
        # Read the content we need to modify
        file_content = self.read_file(path, start_line, end_line).expandtabs()
        new_file_content = file_content.replace(old_str, new_str)

        # Now we need to write back the file
        # Read the file in chunks and write the modified content in the right place
        temp_file = path.with_suffix('.tmp')
        try:
            with path.open('r', newline='') as f_in, temp_file.open('w', newline='') as f_out:
                # Copy lines before our modification
                for _ in range(start_line - 1):
                    line = f_in.readline()
                    if not line:
                        break
                    f_out.write(line)
                
                # Write our modified content
                # Ensure it ends with a newline if the original did
                if new_file_content and not new_file_content.endswith('\n'):
                    new_file_content += '\n'
                f_out.write(new_file_content)
                
                # Skip the lines we modified
                for _ in range(end_line - start_line + 1):
                    if not f_in.readline():
                        break
                
                # Copy the rest of the file
                while True:
                    line = f_in.readline()
                    if not line:
                        break
                    f_out.write(line)
            
            # Save the original content to history (just the modified part)
            self._file_history[path].append(file_content)
            
            # Replace the original file with our temporary file
            temp_file.replace(path)
            
            success_message = f'The file {path} has been edited. '
            success_message += self._make_output(
                new_file_content, f'a snippet of {path}', start_line
            )
            
            if enable_linting:
                lint_results = self._run_linting(file_content, new_file_content, path)
                success_message += '\n' + lint_results + '\n'
            
            success_message += 'Review the changes and make sure they are as expected. Edit the file again if necessary.'
            return CLIResult(
                output=success_message,
                prev_exist=True,
                path=str(path),
                old_content=file_content,
                new_content=new_file_content,
            )
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def view(self, path: Path, view_range: list[int] | None = None) -> CLIResult:
        """
        View the contents of a file or a directory.
        """
        if path.is_dir():
            if view_range:
                raise EditorToolParameterInvalidError(
                    'view_range',
                    view_range,
                    'The `view_range` parameter is not allowed when `path` points to a directory.',
                )

            # First count hidden files/dirs in current directory only
            # -mindepth 1 excludes . and .. automatically
            _, hidden_stdout, _ = run_shell_cmd(
                rf"find -L {path} -mindepth 1 -maxdepth 1 -name '.*'"
            )
            hidden_count = (
                len(hidden_stdout.strip().split('\n')) if hidden_stdout.strip() else 0
            )

            # Then get files/dirs up to 2 levels deep, excluding hidden entries at both depth 1 and 2
            _, stdout, stderr = run_shell_cmd(
                rf"find -L {path} -maxdepth 2 -not \( -path '{path}/\.*' -o -path '{path}/*/\.*' \) | sort",
                truncate_notice=DIRECTORY_CONTENT_TRUNCATED_NOTICE,
            )
            if not stderr:
                msg = [
                    f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}"
                ]
                if hidden_count > 0:
                    msg.append(
                        f"\n{hidden_count} hidden files/directories in this directory are excluded. You can use 'ls -la {path}' to see them."
                    )
                stdout = '\n'.join(msg)
            return CLIResult(
                output=stdout,
                error=stderr,
                path=str(path),
                prev_exist=True,
            )

        if not view_range:
            file_content = self.read_file(path)
            return CLIResult(
                output=self._make_output(file_content, str(path), 1),
                path=str(path),
                prev_exist=True,
            )

        if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
            raise EditorToolParameterInvalidError(
                'view_range',
                view_range,
                'It should be a list of two integers.',
            )

        start_line, end_line = view_range
        if start_line < 1:
            raise EditorToolParameterInvalidError(
                'view_range',
                view_range,
                'Its first element should be greater than or equal to 1.',
            )

        if end_line != -1 and end_line < start_line:
            raise EditorToolParameterInvalidError(
                'view_range',
                view_range,
                f'Its second element `{end_line}` should be greater than or equal to the first element `{start_line}`.',
            )

        # Get the content for the requested range
        file_content = self.read_file(path, start_line, end_line if end_line != -1 else None)
        return CLIResult(
            path=str(path),
            output=self._make_output(file_content, str(path), start_line),
            prev_exist=True,
        )

    def write_file(self, path: Path, file_text: str) -> None:
        """
        Write the content of a file to a given path; raise a ToolError if an error occurs.
        """
        try:
            path.write_text(file_text)
        except Exception as e:
            raise ToolError(f'Ran into {e} while trying to write to {path}') from None

    def insert(
        self, path: Path, insert_line: int, new_str: str, enable_linting: bool
    ) -> CLIResult:
        """
        Implement the insert command, which inserts new_str at the specified line in the file content.
        """
        try:
            file_text = self.read_file(path)
        except Exception as e:
            raise ToolError(f'Ran into {e} while trying to read {path}') from None

        file_text = file_text.expandtabs()
        new_str = new_str.expandtabs()

        file_text_lines = file_text.split('\n')
        num_lines = len(file_text_lines)

        if insert_line < 0 or insert_line > num_lines:
            raise EditorToolParameterInvalidError(
                'insert_line',
                insert_line,
                f'It should be within the range of lines of the file: {[0, num_lines]}',
            )

        new_str_lines = new_str.split('\n')
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_CONTEXT_WINDOW) : insert_line]
            + new_str_lines
            + file_text_lines[
                insert_line : min(num_lines, insert_line + SNIPPET_CONTEXT_WINDOW)
            ]
        )
        new_file_text = '\n'.join(new_file_text_lines)
        snippet = '\n'.join(snippet_lines)

        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        success_message = f'The file {path} has been edited. '
        success_message += self._make_output(
            snippet,
            'a snippet of the edited file',
            max(1, insert_line - SNIPPET_CONTEXT_WINDOW + 1),
        )

        if enable_linting:
            # Run linting on the changes
            lint_results = self._run_linting(file_text, new_file_text, path)
            success_message += '\n' + lint_results + '\n'

        success_message += 'Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary.'
        return CLIResult(
            output=success_message,
            prev_exist=True,
            path=str(path),
            old_content=file_text,
            new_content=new_file_text,
        )

    def validate_path(self, command: Command, path: Path) -> None:
        """
        Check that the path/command combination is valid.
        """
        # Check if its an absolute path
        if not path.is_absolute():
            suggested_path = Path.cwd() / path
            raise EditorToolParameterInvalidError(
                'path',
                path,
                f'The path should be an absolute path, starting with `/`. Maybe you meant {suggested_path}?',
            )
        # Check if path and command are compatible
        if command == 'create' and path.exists():
            raise EditorToolParameterInvalidError(
                'path',
                path,
                f'File already exists at: {path}. Cannot overwrite files using command `create`.',
            )
        if command != 'create' and not path.exists():
            raise EditorToolParameterInvalidError(
                'path',
                path,
                f'The path {path} does not exist. Please provide a valid path.',
            )
        if command != 'view' and path.is_dir():
            raise EditorToolParameterInvalidError(
                'path',
                path,
                f'The path {path} is a directory and only the `view` command can be used on directories.',
            )

    def undo_edit(self, path: Path) -> CLIResult:
        """
        Implement the undo_edit command.
        """
        if not self._file_history[path]:
            raise ToolError(f'No edit history found for {path}.')

        current_text = self.read_file(path).expandtabs()
        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)

        return CLIResult(
            output=f'Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}',
            path=str(path),
            prev_exist=True,
            old_content=current_text,
            new_content=old_text,
        )

    def read_file(self, path: Path, start_line: int = None, end_line: int = None) -> str:
        """
        Read the content of a file from a given path; raise a ToolError if an error occurs.
        If start_line and end_line are provided, only read those lines.
        """
        try:
            if start_line is None and end_line is None:
                # For small files or when we need the whole content, read all at once
                if path.stat().st_size < 1024 * 1024:  # 1MB
                    return path.read_text()
                
                # For large files, read line by line
                content = []
                with path.open() as f:
                    for line in f:
                        content.append(line.rstrip('\n'))
                return '\n'.join(content)
            else:
                # Read specific lines
                content = []
                with path.open() as f:
                    for i, line in enumerate(f, 1):
                        if start_line and i < start_line:
                            continue
                        if end_line and i > end_line:
                            break
                        content.append(line.rstrip('\n'))
                return '\n'.join(content)
        except Exception as e:
            raise ToolError(f'Ran into {e} while trying to read {path}') from None

    def _make_output(
        self,
        snippet_content: str,
        snippet_description: str,
        start_line: int = 1,
        expand_tabs: bool = True,
    ) -> str:
        """
        Generate output for the CLI based on the content of a code snippet.
        """
        snippet_content = maybe_truncate(
            snippet_content, truncate_notice=FILE_CONTENT_TRUNCATED_NOTICE
        )
        if expand_tabs:
            snippet_content = snippet_content.expandtabs()

        snippet_content = '\n'.join(
            [
                f'{i + start_line:6}\t{line}'
                for i, line in enumerate(snippet_content.split('\n'))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {snippet_description}:\n"
            + snippet_content
            + '\n'
        )

    def _run_linting(self, old_content: str, new_content: str, path: Path) -> str:
        """
        Run linting on file changes and return formatted results.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create paths with exact filenames in temp directory
            temp_old = Path(temp_dir) / f'old.{path.name}'
            temp_new = Path(temp_dir) / f'new.{path.name}'

            # Write content to temporary files
            temp_old.write_text(old_content)
            temp_new.write_text(new_content)

            # Run linting on the changes
            results = self._linter.lint_file_diff(str(temp_old), str(temp_new))

            if not results:
                return 'No linting issues found in the changes.'

            # Format results
            output = ['Linting issues found in the changes:']
            for result in results:
                output.append(
                    f'- Line {result.line}, Column {result.column}: {result.message}'
                )
            return '\n'.join(output) + '\n'
