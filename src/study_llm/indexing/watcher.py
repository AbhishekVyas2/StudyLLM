"""
Filesystem watcher for StudyLLM (watchdog-based).
Debounces FS events into enqueue/delete actions.
"""

import logging
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


if WATCHDOG_AVAILABLE:

    class _Handler(FileSystemEventHandler):
        """Collect raw watchdog events, debounce, and dispatch."""

        DEBOUNCE_SECONDS = 0.5

        def __init__(self, data_root: Path, on_index: Callable[[Path], None],
                     on_delete: Callable[[str], None]):
            self.data_root = Path(data_root).resolve()
            self.on_index = on_index
            self.on_delete = on_delete
            self._pending = {}
            self._lock = threading.Lock()
            self._timer = None

        def _schedule(self):
            def fire():
                with self._lock:
                    pending = dict(self._pending)
                    self._pending.clear()
                for rel_path, action in pending.items():
                    try:
                        if action == "delete":
                            self.on_delete(rel_path)
                        else:
                            self.on_index(self.data_root / rel_path)
                    except Exception as e:
                        logger.error(f"Watcher dispatch error: {e}")

            with self._lock:
                if self._timer is not None:
                    self._timer.cancel()
                self._timer = threading.Timer(self.DEBOUNCE_SECONDS, fire)
                self._timer.start()

        def _relpath(self, path) -> str:
            return str(Path(path).relative_to(self.data_root)).replace("\\", "/")

        def on_created(self, event):
            if not event.is_directory:
                self._pending[self._relpath(event.src_path)] = "index"

        def on_modified(self, event):
            if not event.is_directory:
                self._pending[self._relpath(event.src_path)] = "index"

        def on_moved(self, event):
            # Treat moves as: delete source, index destination
            if not event.is_directory:
                try:
                    dest_rel = self._relpath(event.dest_path)
                except ValueError:
                    return
                self._pending[dest_rel] = "index"
                try:
                    src_rel = self._relpath(event.src_path)
                    if src_rel != dest_rel:
                        self._pending[src_rel] = "delete"
                except ValueError:
                    pass

        def on_deleted(self, event):
            if not event.is_directory:
                self._pending[self._relpath(event.src_path)] = "delete"


class Watcher:
    """Watches data/ recursively and triggers indexing/deletion."""

    def __init__(self, data_root: Path, on_index: Callable[[Path], None],
                 on_delete: Callable[[str], None]):
        self.handler = _Handler(data_root, on_index, on_delete)             if WATCHDOG_AVAILABLE else None
        self.observer = None
        self.on_index = on_index
        self.on_delete = on_delete
        self.data_root = Path(data_root).resolve()

    def start(self):
        if not WATCHDOG_AVAILABLE or self.observer is not None:
            logger.warning(
                "watchdog not installed - live updates disabled. "
                "Install with: pip install watchdog"
            )
            return
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.data_root), recursive=True)
        self.observer.start()
        logger.info(f"Watching {self.data_root} for changes")

    def stop(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None
