from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
import traceback
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from app.config import PROJECT_ROOT, Settings
from app.desktop.clipboard import install_clipboard_support_tree
from app.desktop.controller import (
    DEFAULT_INDEX_TARGET,
    ActiveIndexInfo,
    LabController,
    OperationResult,
    PreflightReport,
)
from app.lab.progress import ProgressEvent
from app.lab.resources import ResourceSample
from app.retrieval.vector_store import QdrantBackendStatus
from app.schemas import QueryResponse
from app.wiki.corpus import WikiCorpusConfig, WikiCorpusStats

logger = logging.getLogger(__name__)

STAGE_LABELS = {
    "corpus_discovery": "Scanning corpus",
    "manifest_handling": "Manifest scan",
    "parsing_cleaning_chunking": "Parsing / cleaning / chunking documents",
    "loading_embedding_model": "Loading embedding model",
    "embedding_inference": "Embedding",
    "qdrant_initialization": "Qdrant initialization",
    "qdrant_upsert": "Qdrant upsert",
    "lexical_corpus": "Updating lexical corpus",
    "bm25_build": "Building BM25",
    "finalizing": "Finalizing",
    "index_persisted": "Index persisted",
    "ready": "Ready",
}


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return "N/A"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


class RAGLabApp:
    def __init__(self, root: tk.Tk, settings: Settings) -> None:
        self.root = root
        self.settings = settings
        self.controller = LabController(settings)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.active_operation: str | None = None
        self.operation_started: float | None = None
        self._last_progress_log: tuple[str, str, int | float | None] | None = None
        self.root.title("Document RAG Lab")
        self.root.geometry("1050x800")
        self.root.minsize(850, 650)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.stage_var = tk.StringVar(value="Starting")
        self.elapsed_var = tk.StringVar(value="Elapsed: 0 s")
        self.resource_var = tk.StringVar(value="CPU: N/A | RAM: N/A | GPU: N/A | Disk: N/A")
        self.detail_var = tk.StringVar(value="")
        self.last_benchmark_var = tk.StringVar(value="No benchmark yet")

        self._build_ui()
        install_clipboard_support_tree(self.root)
        self.root.after(100, self._poll_events)
        self.root.after(150, self.refresh_environment)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.setup_tab = ttk.Frame(notebook)
        self.dataset_tab = ttk.Frame(notebook)
        self.index_tab = ttk.Frame(notebook)
        self.chat_tab = ttk.Frame(notebook)
        notebook.add(self.setup_tab, text="Setup / Environment")
        notebook.add(self.dataset_tab, text="Dataset")
        notebook.add(self.index_tab, text="Index")
        notebook.add(self.chat_tab, text="RAG Query")
        self._build_setup_tab()
        self._build_dataset_tab()
        self._build_index_tab()
        self._build_chat_tab()

        status = ttk.LabelFrame(self.root, text="Current operation")
        status.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(status, textvariable=self.stage_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.elapsed_var).grid(row=0, column=1, padx=20)
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=350)
        self.progress.grid(row=0, column=2, sticky="ew", padx=8)
        ttk.Label(status, textvariable=self.detail_var).grid(
            row=1, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(status, textvariable=self.resource_var).grid(
            row=2, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(status, textvariable=self.last_benchmark_var).grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        ttk.Button(status, text="Show benchmark folder", command=self._open_benchmark).grid(
            row=3, column=2, sticky="e"
        )
        status.columnconfigure(2, weight=1)

    def _build_setup_tab(self) -> None:
        frame = ttk.Frame(self.setup_tab, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Button(frame, text="Refresh preflight", command=self.refresh_environment).pack(
            anchor="w"
        )
        self.environment_text = scrolledtext.ScrolledText(frame, height=28, wrap="word")
        self.environment_text.pack(fill="both", expand=True, pady=8)

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Callable[[], None],
        button: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(parent, text=button, command=command).grid(row=row, column=2, pady=4)

    def _build_dataset_tab(self) -> None:
        frame = ttk.Frame(self.dataset_tab, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        self.dump_var = tk.StringVar(value=str(self.settings.wiki_dump_path))
        self.corpus_output_var = tk.StringVar(value=str(self.settings.wiki_corpus_path))
        self._path_row(frame, 0, "Wikipedia XML.BZ2", self.dump_var, self._select_dump, "Select")
        self._path_row(
            frame, 1, "Corpus output", self.corpus_output_var, self._select_output, "Select"
        )
        ttk.Button(frame, text="Download configured dump", command=self._download_dump).grid(
            row=2, column=1, sticky="w", pady=6
        )
        self.target_mode_var = tk.StringVar(value="size")
        ttk.Label(frame, text="Target type").grid(row=3, column=0, sticky="w")
        mode_frame = ttk.Frame(frame)
        mode_frame.grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(
            mode_frame, text="Corpus size", variable=self.target_mode_var, value="size"
        ).pack(side="left")
        ttk.Radiobutton(
            mode_frame, text="Article limit", variable=self.target_mode_var, value="limit"
        ).pack(side="left", padx=12)
        ttk.Label(frame, text="Size preset").grid(row=4, column=0, sticky="w")
        self.target_size_var = tk.StringVar(value="100 MB")
        ttk.Combobox(
            frame,
            textvariable=self.target_size_var,
            values=("100 MB", "500 MB", "1 GB", "2 GB", "8 GB", "Custom"),
            state="readonly",
        ).grid(row=4, column=1, sticky="w", pady=4)
        custom_frame = ttk.Frame(frame)
        custom_frame.grid(row=5, column=1, sticky="w")
        self.custom_size_var = tk.StringVar(value="100")
        ttk.Entry(custom_frame, textvariable=self.custom_size_var, width=10).pack(side="left")
        self.custom_unit_var = tk.StringVar(value="MB")
        ttk.Combobox(
            custom_frame,
            textvariable=self.custom_unit_var,
            values=("MB", "GB"),
            state="readonly",
            width=5,
        ).pack(side="left", padx=4)
        ttk.Label(frame, text="Custom size").grid(row=5, column=0, sticky="w")
        ttk.Label(frame, text="Article limit").grid(row=6, column=0, sticky="w")
        self.article_limit_var = tk.StringVar(value="10")
        ttk.Entry(frame, textvariable=self.article_limit_var, width=16).grid(
            row=6, column=1, sticky="w", pady=4
        )
        ttk.Label(frame, text="Minimum article chars").grid(row=7, column=0, sticky="w")
        self.min_chars_var = tk.StringVar(value="3000")
        ttk.Entry(frame, textvariable=self.min_chars_var, width=16).grid(
            row=7, column=1, sticky="w", pady=4
        )
        self.resume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Resume existing corpus", variable=self.resume_var).grid(
            row=8, column=1, sticky="w", pady=4
        )
        ttk.Button(frame, text="Build PDF Corpus", command=self._build_corpus).grid(
            row=9, column=1, sticky="w", pady=10
        )
        self.dataset_output = scrolledtext.ScrolledText(frame, height=14, wrap="word")
        self.dataset_output.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=8)
        frame.rowconfigure(10, weight=1)

    def _build_index_tab(self) -> None:
        frame = ttk.Frame(self.index_tab, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        self.index_corpus_var = tk.StringVar(value=str(self.settings.wiki_corpus_path))
        self._path_row(
            frame, 0, "Corpus directory", self.index_corpus_var, self._select_index_corpus, "Select"
        )
        ttk.Label(frame, text="Index target").grid(row=1, column=0, sticky="nw", pady=4)
        target_frame = ttk.Frame(frame)
        target_frame.grid(row=1, column=1, columnspan=2, sticky="w")
        self.index_target_var = tk.StringVar(value=DEFAULT_INDEX_TARGET)
        ttk.Radiobutton(
            target_frame,
            text="Isolated benchmark / experiment",
            variable=self.index_target_var,
            value="isolated",
        ).pack(anchor="w")
        ttk.Radiobutton(
            target_frame,
            text="Baseline index",
            variable=self.index_target_var,
            value="baseline",
        ).pack(anchor="w")
        ttk.Label(frame, text="Qdrant backend").grid(row=2, column=0, sticky="nw", pady=4)
        backend_frame = ttk.Frame(frame)
        backend_frame.grid(row=2, column=1, columnspan=2, sticky="w")
        self.qdrant_backend_var = tk.StringVar(value=self.settings.qdrant_mode)
        ttk.Radiobutton(
            backend_frame,
            text="Local",
            variable=self.qdrant_backend_var,
            value="local",
        ).pack(side="left")
        ttk.Radiobutton(
            backend_frame,
            text="Server",
            variable=self.qdrant_backend_var,
            value="server",
        ).pack(side="left", padx=12)
        ttk.Label(frame, text="Server URL").grid(row=3, column=0, sticky="w", pady=4)
        self.qdrant_server_url_var = tk.StringVar(value=self.settings.qdrant_url)
        self.qdrant_server_entry = ttk.Entry(
            frame,
            textvariable=self.qdrant_server_url_var,
        )
        self.qdrant_server_entry.grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=4)
        self.qdrant_check_button = ttk.Button(
            frame,
            text="Check Server",
            command=self._check_qdrant_server,
        )
        self.qdrant_check_button.grid(row=3, column=2, sticky="w", pady=4)
        self.qdrant_status_var = tk.StringVar(value="Not checked")
        ttk.Label(frame, text="Server status").grid(row=4, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.qdrant_status_var).grid(
            row=4, column=1, columnspan=2, sticky="w"
        )
        values = (
            f"Chunk size: {self.settings.chunk_size}\n"
            f"Overlap: {self.settings.chunk_overlap}\n"
            f"Embedding: {self.settings.embedding_model}\n"
            f"Embedding device request: {self.settings.embedding_device or 'auto'}\n"
            f"Reranker: {self.settings.reranker_model}, "
            f"candidates={self.settings.rerank_candidates}"
        )
        ttk.Label(frame, text=values, justify="left").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=8
        )
        self.index_target_details_var = tk.StringVar()
        ttk.Label(
            frame,
            textvariable=self.index_target_details_var,
            justify="left",
        ).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=8
        )
        ttk.Button(frame, text="Build / Update Index", command=self._build_index).grid(
            row=7, column=1, sticky="w", pady=8
        )
        ttk.Button(
            frame,
            text="Activate Existing Index",
            command=self._activate_existing_index,
        ).grid(row=7, column=2, sticky="w", pady=8)
        ttk.Label(
            frame,
            text=(
                "Update uses the manifest and skips unchanged files. "
                "Baseline is never deleted automatically."
            ),
        ).grid(row=8, column=0, columnspan=3, sticky="w")
        self.index_output = scrolledtext.ScrolledText(frame, height=22, wrap="word")
        self.index_output.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=8)
        frame.rowconfigure(9, weight=1)
        self.index_corpus_var.trace_add("write", self._refresh_index_target_details)
        self.index_target_var.trace_add("write", self._refresh_index_target_details)
        self.qdrant_backend_var.trace_add("write", self._qdrant_backend_changed)
        self.qdrant_server_url_var.trace_add("write", self._qdrant_server_url_changed)
        self._qdrant_backend_changed()
        self._refresh_index_target_details()

    def _qdrant_backend_changed(self, *_: object) -> None:
        server = self.qdrant_backend_var.get() == "server"
        state = "normal" if server else "disabled"
        self.qdrant_server_entry.configure(state=state)
        self.qdrant_check_button.configure(state=state)
        self.qdrant_status_var.set("Not checked" if server else "Not applicable")
        self._refresh_index_target_details()

    def _qdrant_server_url_changed(self, *_: object) -> None:
        if self.qdrant_backend_var.get() == "server":
            self.qdrant_status_var.set("Not checked")
        self._refresh_index_target_details()

    def _check_qdrant_server(self) -> None:
        url = self.qdrant_server_url_var.get().strip()

        def action() -> None:
            status = self.controller.check_qdrant_server(url)
            self.events.put(("qdrant_status", status))

        self._run_worker("Check Qdrant Server", action)

    def _show_qdrant_status(self, status: QdrantBackendStatus) -> None:
        if status.available:
            version = status.server_version or "unknown"
            self.qdrant_status_var.set(f"Available | version: {version}")
        else:
            self.qdrant_status_var.set(f"Unavailable | {status.error or status.endpoint}")
        self._refresh_index_target_details()

    def _refresh_index_target_details(self, *_: object) -> None:
        try:
            corpus = Path(self.index_corpus_var.get())
            mode = self.index_target_var.get()
            backend = self.qdrant_backend_var.get()
            target = self.controller.plan_index_target(
                corpus,
                mode,  # type: ignore[arg-type]
                backend,  # type: ignore[arg-type]
                self.qdrant_server_url_var.get(),
            )
            settings = target.settings
            index_root = settings.corpus_path.parent
            qdrant_endpoint = (
                display_path(settings.qdrant_path)
                if settings.qdrant_mode == "local"
                else settings.qdrant_url
            )
            qdrant_target = (
                f"Qdrant path: {qdrant_endpoint}\n"
                if settings.qdrant_mode == "local"
                else f"Qdrant URL: {qdrant_endpoint}\n"
            )
            backend_status = (
                "Advisory: Local Qdrant is intended for small development datasets. "
                "For larger experiments use Qdrant Server."
                if settings.qdrant_mode == "local"
                else f"Status: {self.qdrant_status_var.get()}"
            )
            self.index_target_details_var.set(
                f"Corpus: {display_path(corpus)}\n"
                f"Index target: {target.label}\n"
                f"Index path: {display_path(index_root)}\n"
                f"Qdrant backend: {settings.qdrant_mode.title()}\n"
                + qdrant_target
                + f"Collection: {settings.qdrant_collection}\n"
                + f"Lexical corpus: {display_path(settings.corpus_path)}\n"
                + f"Manifest: {display_path(settings.manifest_path)}\n"
                + backend_status
            )
        except (OSError, ValueError) as error:
            self.index_target_details_var.set(f"Index target unavailable: {error}")

    def _build_chat_tab(self) -> None:
        frame = ttk.Frame(self.chat_tab, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Active RAG index").grid(row=0, column=0, sticky="nw")
        self.active_index_details_var = tk.StringVar(value="No active RAG index selected.")
        ttk.Label(
            frame,
            textvariable=self.active_index_details_var,
            justify="left",
        ).grid(row=0, column=1, columnspan=2, sticky="w", pady=(0, 10))
        ttk.Label(frame, text="Mode").grid(row=1, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="quality")
        ttk.Combobox(
            frame,
            textvariable=self.mode_var,
            values=("fast", "quality"),
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w")
        ttk.Label(frame, text="Question").grid(row=2, column=0, sticky="nw", pady=6)
        self.question_text = tk.Text(frame, height=3, wrap="word")
        self.question_text.grid(row=2, column=1, sticky="ew", pady=6)
        ttk.Button(frame, text="Ask", command=self._ask).grid(row=2, column=2, padx=8)
        ttk.Label(frame, text="Answer").grid(row=3, column=0, sticky="nw")
        self.answer_text = scrolledtext.ScrolledText(frame, height=10, wrap="word")
        self.answer_text.grid(row=3, column=1, columnspan=2, sticky="nsew", pady=4)
        self.answer_text.configure(state="disabled")
        ttk.Label(frame, text="Sources / timings").grid(row=4, column=0, sticky="nw")
        self.sources_text = scrolledtext.ScrolledText(frame, height=12, wrap="word")
        self.sources_text.grid(row=4, column=1, columnspan=2, sticky="nsew", pady=4)
        self.sources_text.configure(state="disabled")
        frame.rowconfigure(3, weight=1)
        frame.rowconfigure(4, weight=1)

    def _run_worker(self, name: str, function: Callable[[], object]) -> None:
        if self.busy:
            messagebox.showinfo("RAG Lab", "Another operation is still running")
            return
        self.busy = True
        self.active_operation = name
        self.operation_started = time.perf_counter()
        self._last_progress_log = None
        self.stage_var.set(name)
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)

        def work() -> None:
            try:
                self.events.put(("result", function()))
            except Exception as error:
                logger.exception("Desktop operation failed")
                self.events.put(("error", (error, traceback.format_exc())))
            finally:
                self.events.put(("done", name))

        threading.Thread(target=work, name=f"rag-lab-{name}", daemon=True).start()

    def _progress_callback(self, event: ProgressEvent) -> None:
        self.events.put(("progress", event))

    def _resource_callback(self, sample: ResourceSample) -> None:
        self.events.put(("resource", sample))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    self._show_progress(payload)  # type: ignore[arg-type]
                elif kind == "resource":
                    self._show_resource(payload)  # type: ignore[arg-type]
                elif kind == "preflight":
                    self._show_preflight(payload)  # type: ignore[arg-type]
                elif kind == "corpus_result":
                    self._show_corpus_result(payload)  # type: ignore[arg-type]
                elif kind == "index_result":
                    self._show_index_result(payload)  # type: ignore[arg-type]
                elif kind == "query_result":
                    self._show_query_result(payload)  # type: ignore[arg-type]
                elif kind == "download_result":
                    self.dump_var.set(str(payload))
                elif kind == "qdrant_status":
                    self._show_qdrant_status(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    error, trace = payload  # type: ignore[misc]
                    output = (
                        self.index_output
                        if self.active_operation == "Build / Update Index"
                        else self.dataset_output
                    )
                    self._append(output, trace)
                    messagebox.showerror("RAG Lab error", str(error))
                elif kind == "done":
                    self.busy = False
                    self.active_operation = None
                    self.operation_started = None
                    self.progress.stop()
                    self.stage_var.set("Ready")
        except queue.Empty:
            pass
        if self.busy and self.operation_started is not None:
            self.elapsed_var.set(f"Elapsed: {time.perf_counter() - self.operation_started:.1f} s")
        self.root.after(100, self._poll_events)

    def _show_progress(self, event: ProgressEvent) -> None:
        label = STAGE_LABELS.get(event.stage, event.stage.replace("_", " ").title())
        self.stage_var.set(label)
        if event.fraction is None:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=100, value=event.fraction * 100)
        details = event.details or {}
        if details.get("elapsed_seconds") is not None:
            self.elapsed_var.set(f"Elapsed: {float(details['elapsed_seconds']):.1f} s")
        detail = self._format_progress_detail(event)
        self.detail_var.set(detail)
        if self.active_operation == "Build / Update Index" and self._should_log_progress(event):
            self._append(self.index_output, f"{label}: {detail}\n")

    @staticmethod
    def _format_progress_detail(event: ProgressEvent) -> str:
        details = event.details or {}
        parts = [event.message] if event.message else []
        if event.current is not None and event.total is not None:
            parts.append(f"{event.current:g} / {event.total:g}")
        labels = {
            "documents_processed": "Documents",
            "documents_total": "Documents total",
            "documents_changed": "Changed",
            "documents_skipped": "Skipped",
            "pages_processed": "Pages",
            "chunks_processed": "Chunks",
            "document_batch": "Document batch",
            "document_batches": "Batches total",
            "embedding_batches_completed": "Embedding batches",
            "chunks_in_batch": "Batch chunks",
            "generated_pdfs": "PDFs",
            "inspected_pages": "Inspected",
        }
        for key, label in labels.items():
            if details.get(key) is not None:
                parts.append(f"{label}: {details[key]}")
        if details.get("corpus_bytes") is not None:
            parts.append(f"Corpus: {format_bytes(details['corpus_bytes'])}")
        if details.get("eta_seconds") is not None:
            parts.append(f"ETA: {float(details['eta_seconds']):.1f} s")
        return " | ".join(parts) or event.kind.title()

    def _should_log_progress(self, event: ProgressEvent) -> bool:
        key = (event.stage, event.kind, event.current)
        if key == self._last_progress_log:
            return False
        if event.kind in {"started", "completed", "message"}:
            self._last_progress_log = key
            return True
        if event.current is None or event.total is None:
            return False
        interval = max(1, int(event.total) // 25)
        should_log = int(event.current) % interval == 0 or event.current == event.total
        if should_log:
            self._last_progress_log = key
        return should_log

    def _show_resource(self, sample: ResourceSample) -> None:
        gpu = (
            f"{sample.gpu_utilization_percent:.0f}% / "
            f"{format_bytes(sample.gpu_vram_used_bytes)}"
            if sample.gpu_utilization_percent is not None
            else "N/A"
        )
        self.resource_var.set(
            f"CPU: {sample.system_cpu_percent}% | "
            f"Process: {sample.process_cpu_percent}% | "
            f"RAM: {sample.system_ram_percent}% | "
            f"RSS: {format_bytes(sample.process_rss_bytes)} | GPU: {gpu} | "
            f"Disk free: {format_bytes(sample.disk_free_bytes)} | "
            f"R/W: {format_bytes(sample.disk_read_bytes_per_second)}/s / "
            f"{format_bytes(sample.disk_write_bytes_per_second)}/s"
        )

    def refresh_environment(self) -> None:
        def action() -> None:
            report = self.controller.preflight(Path(self.corpus_output_var.get()))
            self.events.put(("preflight", report))

        self._run_worker("Preflight", action)

    def _show_preflight(self, report: PreflightReport) -> None:
        env = report.environment
        text = (
            f"Platform: {env.os} {env.os_version}\nArchitecture: {env.architecture}\n"
            f"Hostname: {env.hostname}\nPython: {env.python_version}\nCPU: {env.cpu_name}\n"
            f"CPU cores: {env.physical_cpu_cores} physical / {env.logical_cpu_cores} logical\n"
            f"RAM: {format_bytes(env.total_ram_bytes)} total / "
            f"{format_bytes(env.available_ram_bytes)} available\n"
            f"Filesystem: {env.disk_filesystem or 'N/A'}\n"
            f"Disk free: {format_bytes(env.disk_free_bytes)}\n"
            f"PyTorch: {env.pytorch_version or 'N/A'}\nCUDA: {env.cuda_available}\n"
            f"CUDA device: {env.cuda_device_name or 'N/A'}\nMPS: {env.mps_available}\n"
            f"Embedding device: {report.embedding_device.selected}\n"
            f"Reranker device: {report.reranker_device.selected}\n"
            f"LLM backend: Ollama / {report.configured_ollama_model}\n"
            f"Ollama running: {env.ollama_available}\n"
            f"Model available: {report.ollama_model_available}\n"
            f"Ollama models: {', '.join(env.ollama_models) or 'N/A'}\n"
            f"aria2c: {env.aria2c_available}\nDocker executable: {env.docker_available}\n"
            f"Corpus: {report.corpus['path']} | PDFs={report.corpus['pdf_count']} | "
            f"{format_bytes(report.corpus['bytes'])}\n"
            f"Index: {'exists' if report.index_exists else 'missing'} | "
            f"{format_bytes(report.index_size_bytes)}\n"
            f"Qdrant backend: {report.qdrant.mode.title()}\n"
            f"Qdrant endpoint: {report.qdrant.endpoint}\n"
            f"Qdrant available: {report.qdrant.available}\n"
            f"Qdrant server version: {report.qdrant.server_version or 'N/A'}\n"
            f"Qdrant client version: {report.qdrant.client_version or 'N/A'}\n"
        )
        if report.warnings:
            text += "\nWarnings:\n- " + "\n- ".join(report.warnings)
        self.environment_text.delete("1.0", "end")
        self.environment_text.insert("end", text)

    def _select_dump(self) -> None:
        selected = filedialog.askopenfilename(filetypes=[("Wikipedia dump", "*.bz2")])
        if selected:
            self.dump_var.set(selected)

    def _select_output(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.corpus_output_var.set(selected)
            self.index_corpus_var.set(selected)

    def _select_index_corpus(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.index_corpus_var.set(selected)

    def _download_dump(self) -> None:
        def action() -> None:
            path = self.controller.download_wiki_dump(callback=self._progress_callback)
            self.events.put(("download_result", path))

        self._run_worker("Download", action)

    def _target_bytes(self) -> int:
        selected = self.target_size_var.get()
        if selected == "Custom":
            amount, unit = self.custom_size_var.get(), self.custom_unit_var.get()
        else:
            amount, unit = selected.split()
        multiplier = 1024**2 if unit == "MB" else 1024**3
        return int(float(amount) * multiplier)

    def _build_corpus(self) -> None:
        try:
            target_bytes = self._target_bytes() if self.target_mode_var.get() == "size" else None
            limit = int(self.article_limit_var.get()) if target_bytes is None else None
            output = Path(self.corpus_output_var.get())
            config = WikiCorpusConfig(
                source=Path(self.dump_var.get()),
                output=output,
                limit=limit,
                target_bytes=target_bytes,
                min_article_chars=int(self.min_chars_var.get()),
                font_path=self.settings.pdf_font_path,
                resume=self.resume_var.get(),
            )
            if target_bytes:
                warning = self.controller.disk_advisory(output, target_bytes)
                if warning and not messagebox.askyesno(
                    "Disk capacity warning", warning + "\nContinue?"
                ):
                    return
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid dataset settings", str(error))
            return

        def action() -> None:
            result = self.controller.build_wiki_corpus(
                config, self._progress_callback, self._resource_callback
            )
            self.events.put(("corpus_result", result))

        self._run_worker("Build PDF corpus", action)

    def _show_corpus_result(self, result: OperationResult) -> None:
        stats = result.value
        assert isinstance(stats, WikiCorpusStats)
        self.index_corpus_var.set(self.corpus_output_var.get())
        self._append(
            self.dataset_output,
            f"PDF corpus ready\nPDFs: {stats.generated_pdfs}\n"
            f"Inspected: {stats.inspected_pages}\nSize: {format_bytes(stats.corpus_bytes)}\n"
            f"Elapsed: {stats.elapsed_seconds:.2f} s\nManifest: {stats.manifest_path}\n"
            f"Benchmark: {result.benchmark_path}\n\n",
        )
        self.last_benchmark_var.set(str(result.benchmark_path))

    def _build_index(self) -> None:
        corpus = Path(self.index_corpus_var.get())
        if not corpus.is_dir():
            messagebox.showerror("Corpus", f"Corpus directory does not exist: {corpus}")
            return
        target_mode = self.index_target_var.get()
        baseline_confirmed = False
        if target_mode == "baseline":
            baseline_confirmed = messagebox.askyesno(
                "Baseline index",
                "You are about to modify the baseline index.\nContinue?",
                default=messagebox.NO,
            )
            if not baseline_confirmed:
                return

        def action() -> None:
            result = self.controller.build_index(
                corpus,
                self._progress_callback,
                self._resource_callback,
                target_mode=target_mode,  # type: ignore[arg-type]
                baseline_confirmed=baseline_confirmed,
                qdrant_backend=self.qdrant_backend_var.get(),  # type: ignore[arg-type]
                qdrant_url=self.qdrant_server_url_var.get(),
            )
            self.events.put(("index_result", result))

        self._run_worker("Build / Update Index", action)

    def _activate_existing_index(self) -> None:
        corpus = Path(self.index_corpus_var.get())
        if not corpus.is_dir():
            messagebox.showerror("Corpus", f"Corpus directory does not exist: {corpus}")
            return
        try:
            info = self.controller.activate_index(
                corpus,
                self.index_target_var.get(),  # type: ignore[arg-type]
                self.qdrant_backend_var.get(),  # type: ignore[arg-type]
                self.qdrant_server_url_var.get(),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            messagebox.showerror("Activate RAG index", str(error))
            return
        self._show_active_index(info)
        self._append(
            self.index_output,
            f"Activated {info.target_label}: {display_path(info.index_path)}\n\n",
        )

    def _show_index_result(self, result: OperationResult) -> None:
        stats = result.value
        assert hasattr(stats, "timings")
        summary = result.summary
        qdrant_seconds = float(summary.get("qdrant_initialization_seconds", 0)) + float(
            summary.get("qdrant_upsert_seconds", 0)
        )
        lines = [
            "\nIndex ready",
            "Corpus",
            f"PDFs: {summary.get('files_discovered', 'N/A')}",
            f"Pages: {summary.get('pages_parsed', 'N/A')}",
            f"Chunks: {summary.get('chunks_created', 'N/A')}",
            "",
            "Build",
            f"Parsing: {float(summary.get('parsing_seconds', 0)):.3f} s",
            f"Cleaning: {float(summary.get('cleaning_seconds', 0)):.3f} s",
            f"Chunking: {float(summary.get('chunking_seconds', 0)):.3f} s",
            f"Model loading: {float(summary.get('embedding_model_loading_seconds', 0)):.3f} s",
            f"Embeddings: {float(summary.get('embedding_seconds', 0)):.3f} s",
            f"Qdrant: {qdrant_seconds:.3f} s",
            f"BM25: {float(summary.get('bm25_build_seconds', 0)):.3f} s",
            f"Total: {float(summary.get('total_seconds', 0)):.3f} s",
            "",
            "Performance",
            f"Pages/sec: {float(summary.get('pages_per_second', 0)):.2f}",
            f"Chunks/sec: {float(summary.get('chunks_per_second', 0)):.2f}",
            "",
            "Resources (peaks)",
            f"Process RSS: {format_bytes(summary.get('peak_process_rss_bytes'))}",
            f"System RAM used: {format_bytes(summary.get('peak_system_ram_used_bytes'))}",
            f"System RAM: {summary.get('peak_system_ram_percent', 'N/A')}%",
            f"System CPU: {summary.get('peak_system_cpu_percent', 'N/A')}%",
            f"Process CPU: {summary.get('peak_process_cpu_percent', 'N/A')}%",
            f"GPU: {summary.get('peak_gpu_utilization_percent', 'N/A')}%",
            f"VRAM: {format_bytes(summary.get('peak_gpu_vram_used_bytes'))}",
            f"GPU temperature: {summary.get('peak_gpu_temperature_c', 'N/A')} C",
            f"GPU power: {summary.get('peak_gpu_power_draw_w', 'N/A')} W",
            "",
            "Storage",
            f"Index size: {format_bytes(summary.get('final_index_size_bytes'))}",
            f"Disk free before: {format_bytes(summary.get('free_disk_before_bytes'))}",
            f"Disk free after: {format_bytes(summary.get('free_disk_after_bytes'))}",
            f"Approx system read: {format_bytes(summary.get('disk_read_delta_bytes'))}",
            f"Approx system write: {format_bytes(summary.get('disk_write_delta_bytes'))}",
            f"Peak read rate: "
            f"{format_bytes(summary.get('peak_disk_read_bytes_per_second'))}/s",
            f"Peak write rate: "
            f"{format_bytes(summary.get('peak_disk_write_bytes_per_second'))}/s",
            f"Benchmark: {result.benchmark_path}",
        ]
        self._append(self.index_output, "\n".join(lines) + "\n\n")
        self.last_benchmark_var.set(str(result.benchmark_path))
        info = self.controller.active_index_info()
        if info is not None:
            self._show_active_index(info)

    def _show_active_index(self, info: ActiveIndexInfo) -> None:
        qdrant_endpoint = (
            display_path(info.qdrant_path)
            if info.qdrant_backend == "local"
            else info.qdrant_endpoint
        )
        indexed_chunks = (
            ">20,000"
            if info.indexed_chunks and info.indexed_chunks > 20_000
            else info.indexed_chunks or "N/A"
        )
        self.active_index_details_var.set(
            f"Active corpus: {display_path(info.corpus_path)}\n"
            f"Active index: {display_path(info.index_path)}\n"
            f"Qdrant backend: {info.qdrant_backend.title()}\n"
            f"Qdrant: {qdrant_endpoint}\n"
            f"Collection: {info.qdrant_collection}\n"
            f"Indexed chunks: {indexed_chunks}\n"
            f"Lexical corpus: {display_path(info.lexical_corpus_path)}\n"
            f"Manifest: {display_path(info.manifest_path)}\n"
            f"Index target: {info.target_label}\n"
            f"Embedding: {info.embedding_model} / {info.embedding_device}\n"
            f"Reranker: {info.reranker_model}, candidates={info.rerank_candidates}\n"
            f"Ollama: {info.ollama_model}"
            + (f"\nAdvisory: {info.qdrant_advisory}" if info.qdrant_advisory else "")
        )

    def _ask(self) -> None:
        query = self.question_text.get("1.0", "end").strip()
        if not query:
            messagebox.showerror("Question", "Enter a question")
            return
        if self.controller.active_index_info() is None:
            messagebox.showerror("RAG index", "No active RAG index selected.")
            return
        mode = self.mode_var.get()

        def action() -> None:
            result = self.controller.ask(
                query, mode, self._progress_callback, self._resource_callback  # type: ignore[arg-type]
            )
            self.events.put(("query_result", result))

        self._run_worker("RAG query", action)

    def _show_query_result(self, result: OperationResult) -> None:
        response = result.value
        assert isinstance(response, QueryResponse)
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", "end")
        self.answer_text.insert("end", response.answer)
        self.answer_text.configure(state="disabled")
        source_lines = [
            f"mode: {result.summary.get('mode', self.mode_var.get())}",
            f"is_answerable: {response.is_answerable}",
        ]
        for index, source in enumerate(response.sources, start=1):
            source_lines.append(
                f"{index}. {source.filename}, page={source.page}, chunk={source.chunk_id}\n"
                f"   dense={source.dense_score} bm25={source.bm25_score} "
                f"rrf={source.rrf_score} reranker={source.reranker_score}"
            )
        timings = response.timings
        source_lines.extend(
            [
                "",
                f"Base retrieval: {timings.base_retrieval_ms:.1f} ms",
                f"Reranker: {timings.reranker_ms:.1f} ms",
                f"Retrieval total: {timings.retrieval_total_ms:.1f} ms",
                f"Generation: {timings.generation_ms:.1f} ms",
                f"Total: {timings.total_ms:.1f} ms",
                f"Context: {timings.context_chars} chars",
                f"Benchmark: {result.benchmark_path}",
            ]
        )
        self.sources_text.configure(state="normal")
        self.sources_text.delete("1.0", "end")
        self.sources_text.insert("end", "\n".join(source_lines))
        self.sources_text.configure(state="disabled")
        self.last_benchmark_var.set(str(result.benchmark_path))

    @staticmethod
    def _append(widget: tk.Text, text: str) -> None:
        widget.insert("end", text)
        widget.see("end")

    def _open_benchmark(self) -> None:
        try:
            self.controller.open_benchmark_folder()
        except Exception as error:
            messagebox.showerror("Benchmark folder", str(error))

    def _close(self) -> None:
        self.controller.close()
        self.root.destroy()


def launch(settings: Settings) -> None:
    root = tk.Tk()
    RAGLabApp(root, settings)
    root.mainloop()
