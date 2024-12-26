# run_capture.py
from pathlib import Path
import time
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from io import StringIO


@dataclass
class RunPaths:
    """Holds paths for a single run"""

    run_dir: Path
    terminal_output: Path
    html_output: Path


# Global context variable to hold the current run paths
current_run_paths: ContextVar[RunPaths] = ContextVar("current_run_paths")


def create_run_paths(
    example_dir_name: str,
) -> RunPaths:
    """Create timestamped run directory and return paths"""
    base_path = Path("concordia_demo") / "examples" / example_dir_name / "saved_runs"
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    run_dir = base_path / f"{timestamp}-run"
    run_dir.mkdir()

    return RunPaths(
        run_dir=run_dir,
        terminal_output=run_dir / "terminal_output.txt",
        html_output=run_dir / "index.html",
    )


@contextlib.contextmanager
def capture_run():
    output = StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        yield output
    # Now, output.getvalue() contains all the captured output.


# @contextlib.contextmanager
# def capture_run(paths: RunPaths):
#     try:
#         with open(paths.terminal_output, "w") as terminal_file:
#             with contextlib.redirect_stdout(terminal_file), contextlib.redirect_stderr(
#                 terminal_file
#             ):
#                 yield
#     finally:
#         pass


class HTMLWriter:
    """Manages HTML file creation and section addition."""

    def __init__(self, paths: RunPaths):
        self.output_path = paths.html_output
        self._write_header()

    def _write_header(self):
        """Write initial HTML structure."""
        header = """<!DOCTYPE html>
<html>
<head>
    <title>Concordia Brainstorm Run</title>
    <style>
        section {
            margin: 20px;
            padding: 20px;
            border: 1px solid #ccc;
        }
        section h2 {
            margin-top: 0;
        }
    </style>
</head>
<body>
"""
        self.output_path.write_text(header)

    def add_section(self, content: str):
        """Add a new section to the HTML file."""
        section = f"""<section>
    {content}
</section>
"""
        with open(self.output_path, "a") as f:
            f.write(section)

    def finalize(self):
        """Close the HTML document."""
        with open(self.output_path, "a") as f:
            f.write("\n</body>\n</html>")


def get_execution_path(example_name: str):
    exec_path = Path(__file__).parent / example_name / f"{example_name}.py"
    print(exec_path)
    return exec_path


if __name__ == "__main__":
    exec_path = get_execution_path("brainstorm")
    # exec_path = (
    #     "/home/avi/iu_esoteric_lab/concordia_demo/concordia_demo/examples/test.py"
    # )
    capture_paths = create_run_paths("brainstorm")

    with capture_run() as output:
        # Read and execute the script content directly
        with open(exec_path) as f:
            code = f.read()
            exec_globals = {
                "capture_paths": capture_paths,
                "__name__": "__main__",  # Ensure code under 'if __name__ == "__main__":' executes
            }
            try:
                exec(code, exec_globals)
            except Exception as e:
                print("script did not complete")
                print(e)

            # Write IO
            with open(capture_paths.terminal_output, "w") as terminal_file:
                terminal_file.write(output.getvalue())

    # with capture_run(capture_paths):
    #     # Read and execute the script content directly
    #     with open(exec_path) as f:
    #         exec(f.read(), {"capture_paths": capture_paths})

    # with capture_run("brainstorm"):
    #     # Get the current context with our run paths
    #     ctx = copy_context()
    #     # Run the script with this context
    #     ctx.run(runpy.run_path, str(exec_path), run_name="__main__")


# class OutputCapture:
#     """Captures terminal output to a file."""

#     def __init__(self, output_path: Path):
#         self.output_path = output_path

#     @contextlib.contextmanager
#     def capture(self):
#         """Context manager to capture terminal output to file."""
#         with open(self.output_path, "w") as terminal_file:
#             with contextlib.redirect_stdout(terminal_file), contextlib.redirect_stderr(
#                 terminal_file
#             ):
#                 yield


# class RunCapture:
#     """Creates and manages a timestamped run directory."""

#     def __init__(self, example_dir_name: str = "brainstorm"):
#         self.base_path = saved_runs_path(example_dir_name)
#         self.base_path.mkdir(parents=True, exist_ok=True)

#         # Create timestamp-based run directory
#         timestamp = int(time.time())
#         self.run_dir = self.base_path / f"{timestamp}-run"
#         self.run_dir.mkdir()

#         # Initialize file paths
#         self.terminal_output = self.run_dir / "terminal_output.txt"
#         self.index_html = self.run_dir / "index.html"

#         # Initialize components
#         self.html_writer = HTMLWriter(self.index_html)
#         self.output_capture = OutputCapture(self.terminal_output)


# def run_brainstorm():
#     """Run the brainstorm script with output capture and HTML generation."""
#     run_dir = RunCapture()

#     # Run the script with output capture
#     with run_dir.output_capture.capture():
#         # This will populate html_content with the display.HTML() calls
#         runpy.run_module(
#             "concordia_demo.examples.brainstorm.brainstorm", run_name="__main__"
#         )

#     # Finalize HTML
#     run_dir.html_writer.finalize()


# if __name__ == "__main__":
#     run_brainstorm()

# class RunDirectory:
#     """Manages output directory structure and file handling for a single run."""

#     def __init__(self, base_dir: str = "concordia_demo/examples/brainstorm/saved_runs"):
#         self.base_path = Path(base_dir)
#         self.base_path.mkdir(parents=True, exist_ok=True)

#         # Create timestamp-based run directory
#         timestamp = int(time.time())
#         self.run_dir = self.base_path / f"{timestamp}-run"
#         self.run_dir.mkdir()

#         # Initialize file paths
#         self.terminal_output = self.run_dir / "terminal_output.txt"
#         self.index_html = self.run_dir / "index.html"

#         # Initialize HTML content
#         self._write_html_header()

#     def _write_html_header(self):
#         """Write initial HTML structure."""
#         header = """<!DOCTYPE html>
# <html>
# <head>
#     <title>Concordia Brainstorm Run</title>
#     <style>
#         section {
#             margin: 20px;
#             padding: 20px;
#             border: 1px solid #ccc;
#         }
#         section h2 {
#             margin-top: 0;
#         }
#     </style>
# </head>
# <body>
# """
#         self.index_html.write_text(header)

#     def add_html_section(self, content: str, title: str):
#         """Add a new section to the HTML file."""
#         section = f"""<section>
#     <h2>{title}</h2>
#     {content}
# </section>
# """
#         with open(self.index_html, "a") as file:
#             file.write(section)

#     def finalize_html(self):
#         """Close the HTML document."""
#         with open(self.index_html, "a") as file:
#             file.write("\n</body>\n</html>")

#     @contextlib.contextmanager
#     def capture_terminal_output(self):
#         """Context manager to capture terminal output to file."""
#         with open(self.terminal_output, "w") as terminal_file:
#             with contextlib.redirect_stdout(terminal_file), contextlib.redirect_stderr(
#                 terminal_file
#             ):
#                 yield


# # Usage example:
# def run_brainstorm():
#     run_dir = RunDirectory()

#     with run_dir.capture_terminal_output():
#         # Your existing code here, but instead of display.HTML(), use:

#         # After first conversation
#         run_dir.add_html_section(first_convo_html, "First Conversation")

#         # After essays
#         run_dir.add_html_section(first_essays_html, "First Essays")

#         # After second conversation
#         run_dir.add_html_section(second_convo_html, "Second Conversation")

#         # After final results
#         run_dir.add_html_section(results_html, "Final Results")

#         # Any print statements or errors will go to terminal_output.txt

#     run_dir.finalize_html()


# if __name__ == "__main__":
#     run_brainstorm()
