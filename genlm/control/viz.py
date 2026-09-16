import webbrowser
import http.server
import socketserver
import threading
import tempfile
import shutil
from importlib.resources import files
from pathlib import Path


class InferenceVisualizer:
    """Web-based visualization server for SMC inference results.

    This class is intended to be used in conjunction with the `InferenceEngine` class.

    Example:
        ```python
        from genlm.control import InferenceVisualizer
        # create the visualizer
        viz = InferenceVisualizer()
        # run inference and save the record to a JSON file
        sequences = await token_sampler.smc(
            n_particles=10,
            max_tokens=20,
            ess_threshold=0.5,
            json_path="smc_record.json" # save the record to a JSON file
        )
        # visualize the inference run
        viz.visualize("smc_record.json")
        # clean up visualization server
        viz.shutdown_server()
        ```
    """

    def __init__(self, port=8000, serve_dir=None):
        """Initialize the visualization server.

        Args:
            port (int): Port to run the server on.
            serve_dir (str | Path, optional): Directory to serve files from.
                If None, creates a temporary directory.

        Raises:
            OSError: If the port is already in use
        """
        self._server = None
        self._server_thread = None
        self._port = port
        # The viewer is llamppl's: it renders the record llamppl's SMCRecord writes.
        self._html_dir = Path(str(files("llamppl") / "html"))

        # Set up serve directory
        if serve_dir is None:
            self._serve_dir = Path(tempfile.mkdtemp(prefix="smc_viz_"))
            self._using_temp_dir = True
        else:
            self._serve_dir = Path(serve_dir).resolve()
            self._using_temp_dir = False
            self._serve_dir.mkdir(exist_ok=True)

        # Create handler that serves from both directories
        class Handler(http.server.SimpleHTTPRequestHandler):
            def translate_path(self_, path):
                # Remove query parameters for file lookup
                clean_path = path.split("?")[0]
                # HTML files come from llamppl's package
                if clean_path.endswith(".html"):
                    return str(self._html_dir / clean_path.lstrip("/"))
                # JSON files come from serve directory
                return str(self._serve_dir / clean_path.lstrip("/"))

        self._start_server(Handler)

    def visualize(self, json_path, auto_open=False):
        """Visualize the inference run in a browser.

        Args:
            json_path (str | Path): Path to the JSON file to visualize. If the file is not
                in the serve directory, it will be copied there. For efficiency, you can
                write JSON files directly to the serve directory
            auto_open (bool): Whether to automatically open in browser

        Returns:
            (str): URL where visualization can be accessed
        """
        if self._server is None:
            raise RuntimeError("Server is not running")

        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # If file isn't in serve directory, copy it there
        dest_path = self._serve_dir / json_path.name
        if json_path.resolve() != dest_path.resolve():
            shutil.copy2(json_path, dest_path)

        url = f"http://localhost:{self._port}/smc.html?path={json_path.name}"

        if auto_open:
            webbrowser.open(url)

        return url

    def _start_server(self, handler_class):
        """Start the HTTP server."""
        try:
            self._server = socketserver.TCPServer(
                ("", self._port), handler_class, bind_and_activate=False
            )
            self._server.allow_reuse_address = True
            self._server.server_bind()
            self._server.server_activate()
        except OSError as e:
            if e.errno == 48 or e.errno == 98:  # Address already in use
                raise OSError(f"Port {self._port} is already in use") from None
            raise

        self._server_thread = threading.Thread(target=self._server.serve_forever)
        self._server_thread.daemon = True
        self._server_thread.start()

    def shutdown_server(self):
        """Shut down the visualization server."""
        if self._server is not None:
            if self._server_thread is not None and self._server_thread.is_alive():
                self._server.shutdown()
                self._server_thread.join()
            self._server.server_close()
            self._server = None
            self._server_thread = None

        # Clean up any temporary files
        if self._using_temp_dir and self._serve_dir.exists():
            shutil.rmtree(self._serve_dir)

    def __del__(self):
        """Ensure server is shut down when object is deleted."""
        self.shutdown_server()
