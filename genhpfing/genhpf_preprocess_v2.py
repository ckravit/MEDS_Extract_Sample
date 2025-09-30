#!/usr/bin/env python3
"""
Standalone GenHPF MEDS Preprocessing Script with Enhanced Monitoring
Handles resource management, progress tracking, and robust error handling
"""

import subprocess
import logging
import psutil
import time
import os
import sys
import signal
import glob
import h5py
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import math
import shutil
import re
try:
    import pyarrow.parquet as pq  # for fast row counts without reading data
except Exception:
    pq = None


# --- ETA configuration ---
ETA_MODE = "dynamic"  # options: "dynamic", "size", "off"
ETA_THROUGHPUT_CODES_PER_SEC = 150.0  # used only if ETA_MODE == "size"


class GenHPFPreprocessor:
    def __init__(self,
                 meds_data_dir: str,
                 meds_labels_dir: str,
                 meds_metadata_dir: str,
                 meds_output_dir: str,
                 max_event_length: int = 256,
                 log_dir: str = "logs"):

        self.meds_data_dir = meds_data_dir
        self.meds_labels_dir = meds_labels_dir
        self.meds_metadata_dir = meds_metadata_dir
        self.meds_output_dir = meds_output_dir
        self.max_event_length = max_event_length
        self.log_dir = Path(log_dir)

        # Create log directory
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging with rotation
        self.setup_logging()

        # Resource monitoring
        self.start_time = None
        self.last_progress_time = None
        self._last_process_log_size = 0
        self._seen_h5_count = 0
        self._process_log_path: Optional[Path] = None

    def setup_logging(self):
        """Setup comprehensive logging with file rotation"""
        log_file = self.log_dir / f"genhpf_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def get_optimal_workers(self, max_workers: int = None) -> int:
        """Calculate optimal number of workers based on system resources"""
        # Get system info
        cpu_count = psutil.cpu_count(logical=True)
        available_gb = psutil.virtual_memory().available / (1024**3)

        # Conservative estimate: 6GB per worker for MEDS processing
        memory_based_workers = max(1, int(available_gb / 6))
        cpu_based_workers = max(1, max(1, cpu_count - 12))  # Leave 12 cores for system

        optimal_workers = min(memory_based_workers, cpu_based_workers)

        if max_workers:
            optimal_workers = min(optimal_workers, max_workers)

        self.logger.info(f"System resources: {cpu_count} CPUs, {available_gb:.1f}GB available memory")
        self.logger.info(f"Calculated optimal workers: {optimal_workers}")

        return optimal_workers

    def monitor_resources(self):
        """Log current system resource usage (uses the output dir mount for disk stats)"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Measure free space where output will be written
        try:
            disk = psutil.disk_usage(self.meds_output_dir)
        except Exception:
            disk = psutil.disk_usage('/')

        self.logger.info(
            "Resources - CPU: %.1f%%, Memory: %.1f%% (%.1fGB free), Disk: %.1f%% (%.1fGB free)",
            cpu_percent,
            memory.percent,
            memory.available/1024**3,
            disk.percent,
            disk.free/1024**3
        )

    def check_prerequisites(self) -> bool:
        """Verify all required paths and dependencies exist"""
        self.logger.info("Checking prerequisites...")

        # Check paths
        paths_to_check = [
            (self.meds_data_dir, "MEDS data directory"),
            (self.meds_labels_dir, "MEDS labels directory"),
            (self.meds_metadata_dir, "MEDS metadata directory")
        ]

        for path, description in paths_to_check:
            if not Path(path).exists():
                self.logger.error(f"{description} does not exist: {path}")
                return False
            self.logger.info(f"✓ {description}: {path}")

        # Check for codes.parquet
        codes_file = Path(self.meds_metadata_dir) / "codes.parquet"
        if not codes_file.exists():
            self.logger.error(f"codes.parquet not found in metadata directory: {codes_file}")
            return False
        self.logger.info(f"✓ codes.parquet found: {codes_file}")

        # Check GenHPF installation
        try:
            result = subprocess.run(['genhpf-preprocess-meds', '--help'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.logger.error("genhpf-preprocess-meds not accessible")
                return False
            self.logger.info("✓ GenHPF installation verified")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.error("genhpf-preprocess-meds command not found or not responding")
            return False

        return True

    def estimate_processing_time(self, num_workers: int) -> str:
        """
        Estimate processing time before we launch the subprocess.
        Prefer Parquet/CSV **row counts** over file size, then convert to hours with a tunable rows/sec/worker heuristic.
        """
        try:
            # Gather files
            data_files: List[str] = []
            data_path = Path(self.meds_data_dir)
            if data_path.is_dir():
                data_files.extend(glob.glob(str(data_path / "**/*.parquet"), recursive=True))
                data_files.extend(glob.glob(str(data_path / "**/*.csv"), recursive=True))
            else:
                data_files = [str(data_path)]

            total_rows = 0
            parquet_rows = 0
            csv_rows = 0

            # --- Prefer Parquet metadata row counts (fast; no data read) ---
            if pq:
                for f in data_files:
                    if f.endswith(".parquet"):
                        try:
                            parquet_rows += pq.ParquetFile(f).metadata.num_rows
                        except Exception:
                            pass

            # --- For CSVs, use wc -l if available; otherwise quick Python line counting ---
            csv_files = [f for f in data_files if f.endswith(".csv")]
            if csv_files:
                wc_exists = shutil.which("wc") is not None
                for f in csv_files:
                    try:
                        if wc_exists:
                            # subtract 1 for header if present (cheap & good enough)
                            out = subprocess.run(["wc", "-l", f], capture_output=True, text=True)
                            n = int(out.stdout.strip().split()[0])
                            csv_rows += max(0, n - 1)
                        else:
                            # fallback: iterate with minimal memory
                            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                                # subtract header if present (best effort)
                                n = sum(1 for _ in fh)
                                csv_rows += max(0, n - 1)
                    except Exception:
                        # if any CSV count fails, ignore that file for estimate
                        pass

            total_rows = parquet_rows + csv_rows

            # If no rows could be counted, fallback to size heuristic so we always print something
            if total_rows == 0:
                total_size_gb = sum(Path(f).stat().st_size for f in data_files) / (1024**3)
                hours_per_gb = 1.0  # fallback heuristic
                hours = max(0.5, (total_size_gb * hours_per_gb) / max(1, num_workers))
                eta_msg = f"{hours:.1f} hours (size-based fallback)"
                self.logger.info(f"Data size: {total_size_gb:.2f} GB across {len(data_files)} files")
                self.logger.info(f"Estimated processing time: {eta_msg} with {num_workers} workers")
                return eta_msg

            # ---- Convert rows -> hours with tunables ----
            # Start conservative; after a run, you can tighten these two numbers based on your logs.
            ROWS_PER_SEC_PARQUET_PER_WORKER = 2000.0
            ROWS_PER_SEC_CSV_PER_WORKER = 1200.0

            rows_per_sec = 0.0
            if parquet_rows:
                rows_per_sec += ROWS_PER_SEC_PARQUET_PER_WORKER
            if csv_rows:
                # if both parquet & csv exist, treat as weighted average by fraction of rows
                frac_csv = csv_rows / total_rows if total_rows else 0.0
                frac_parquet = 1.0 - frac_csv
                # blend rates
                blended = (ROWS_PER_SEC_PARQUET_PER_WORKER * frac_parquet) + (ROWS_PER_SEC_CSV_PER_WORKER * frac_csv)
                rows_per_sec = blended if rows_per_sec == 0.0 else blended  # simple override for clarity
            # multiply by workers
            rows_per_sec *= max(1, num_workers)

            # guard-rail
            rows_per_sec = max(1.0, rows_per_sec)

            est_seconds = total_rows / rows_per_sec
            hours = est_seconds / 3600.0

            # Pretty print & also include an ETA wall-clock time
            eta_wallclock = (datetime.now() + timedelta(seconds=est_seconds)).strftime("%Y-%m-%d %H:%M")
            self.logger.info(f"Data rows: {total_rows:,}  (parquet={parquet_rows:,}, csv={csv_rows:,})")
            self.logger.info(f"Estimated processing time: {hours:.1f} hours with {num_workers} workers (ETA ~ {eta_wallclock})")
            return f"{hours:.1f} hours (ETA ~ {eta_wallclock})"

        except Exception as e:
            self.logger.warning(f"Could not estimate processing time: {e}")
            return "unknown"

    def _infer_total_codes(self) -> Optional[int]:
        """
        Try to infer total 'work units' = sum of list lengths of 'code' from Parquet.
        Falls back to None if inputs are CSV or on any error.
        """
        try:
            import polars as pl
            p = Path(self.meds_data_dir)
            if p.is_file() and p.suffix == ".parquet":
                return int(
                    pl.scan_parquet(str(p))
                    .select(pl.col("code").list.len().sum().alias("n"))
                    .collect(streaming=True)["n"][0]
                )
            if p.is_dir():
                # Sum across all Parquet files; skip CSV
                files = list(p.rglob("*.parquet"))
                if not files:
                    return None
                lf = pl.concat([pl.scan_parquet(str(f)) for f in files], how="diagonal")
                return int(
                    lf.select(pl.col("code").list.len().sum().alias("n"))
                    .collect(streaming=True)["n"][0]
                )
            return None
        except Exception:
            return None

    def _dynamic_eta_update(self, total_units: Optional[int]) -> None:
        """
        Parse the tail of the GenHPF process log for tqdm lines like:
        'Tokenizing worker-0 batch 0: ... 12345/1780705 [.., 159.09it/s]'
        Then log a rolling ETA.
        """
        if not self._process_log_path or not self._process_log_path.exists():
            return
        try:
            import re
            # Read just the tail to keep it cheap
            with open(self._process_log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 200_000))  # ~last 200KB
                tail = f.read().decode(errors="ignore")
            # Prefer the last occurrence
            m = None
            for match in re.finditer(r"(\d+)\s*/\s*(\d+).*?([\d.]+)\s*it/s", tail):
                m = match
            if not m:
                # Try without the it/s group
                for match in re.finditer(r"(\d+)\s*/\s*(\d+)", tail):
                    m = match
            if not m:
                return
            processed = int(m.group(1))
            total = int(m.group(2))
            now = datetime.now()
            # Keep rolling rate if tqdm 'it/s' is missing
            it_per_s = None
            if m.lastindex and m.lastindex >= 3:
                try:
                    it_per_s = float(m.group(3))
                except Exception:
                    it_per_s = None

            # Smooth rate with our own rolling window
            last = getattr(self, "_eta_last", None)
            if last and not it_per_s:
                dp = max(0, processed - last["processed"])
                dt = (now - last["time"]).total_seconds() or 1.0
                it_per_s = dp / dt if dp > 0 else last.get("rate", 0.0)
            if not it_per_s or it_per_s <= 0:
                return

            self._eta_last = {"processed": processed, "time": now, "rate": it_per_s}

            # If we know a more accurate total from Parquet, prefer it
            if total_units and total_units > total:
                total = total_units

            remaining = max(0, total - processed)
            sec = remaining / it_per_s
            hrs = sec / 3600.0
            self.logger.info(f"ETA (dynamic): ~{hrs:.2f} hours remaining "
                            f"at ~{it_per_s:.0f} it/s (processed {processed:,}/{total:,})")
        except Exception:
            pass

    def _parse_tqdm_eta_from_log(self, log_path: str):
        """
        Peek at the tail of the child log and extract tqdm's remaining time.
        Returns a string like '36:54 remaining (ETA ~ 2025-09-29 22:10)' or None.
        """
        try:
            with open(log_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 200_000), os.SEEK_SET)  # read last ~200 KB
                chunk = f.read().decode('utf-8', errors='ignore')
            # tqdm progress updates often use carriage returns; grab the last segment
            tail = chunk.split('\r')[-1]
            m = re.search(r"\[(?:\d+:\d+:\d+|\d+:\d+)<(?P<remain>\d+:\d+:\d+|\d+:\d+),", tail)
            if not m:
                return None
            remain = m.group('remain')
            parts = [int(p) for p in remain.split(':')]
            secs = parts[0]*3600 + parts[1]*60 + (parts[2] if len(parts) == 3 else 0)
            eta = (datetime.now() + timedelta(seconds=secs)).strftime("%Y-%m-%d %H:%M")
            return f"{remain} remaining (ETA ~ {eta})"
        except Exception:
            return None 


    def _count_output_h5(self) -> int:
        """Count .h5 files under expected split dirs"""
        output_path = Path(self.meds_output_dir)
        expected_dirs = ['held_out', 'train', 'tuning']
        count = 0
        for d in expected_dirs:
            p = output_path / d
            if p.exists():
                count += len(list(p.glob("*.h5")))
        return count

    def _update_progress_marker_if_output_advances(self):
        """Update last_progress_time if we see process log growth or new .h5 files"""
        # Log growth
        try:
            sz = self._process_log_path.stat().st_size if self._process_log_path else 0
        except Exception:
            sz = 0

        grew = sz > self._last_process_log_size
        if grew:
            self._last_process_log_size = sz
            self.last_progress_time = datetime.now()

        # New h5 files
        try:
            h5_count = self._count_output_h5()
            if h5_count > self._seen_h5_count:
                self._seen_h5_count = h5_count
                self.last_progress_time = datetime.now()
        except Exception:
            pass

    def _popen_with_own_group(self, command, stdout_handle=None, env=None):
        """Start subprocess in its own process group for group termination"""
        if os.name == "nt":
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            return subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=CREATE_NEW_PROCESS_GROUP,
                env=env,
            )
        else:
            return subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,   # or start_new_session=True on 3.8+
                env=env,
            )


    # def _terminate_process_group(self, process: subprocess.Popen):
    #     """Terminate the entire child process group"""
    #     try:
    #         if os.name == "nt":
    #             # send CTRL-BREAK to the process group; fall back to terminate
    #             try:
    #                 process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
    #             except Exception:
    #                 process.terminate()
    #         else:
    #             os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    #     except Exception:
    #         try:
    #             process.terminate()
    #         except Exception:
    #             pass

    def _write_pid_files(self, pid: int):
        pgid = os.getpgid(pid)
        state_dir = self.log_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        pid_path  = state_dir / "genhpf_current.pid"
        pgid_path = state_dir / "genhpf_current.pgid"
        pid_path.write_text(str(pid))
        pgid_path.write_text(str(pgid))
        self.logger.info(f"Child PID={pid}, PGID={pgid} (saved to {pid_path} / {pgid_path})")

    def _remove_pid_files(self):
        for name in ("genhpf_current.pid", "genhpf_current.pgid"):
            p = self.log_dir / name
            try:
                p.unlink()
            except FileNotFoundError:
                pass


    def _terminate_process_group(self, process: subprocess.Popen, reason: str = "user request"):
        """
        Best-effort graceful -> forceful teardown of the whole process group.
        """
        if process.poll() is not None:
            return  # already exited

        pgid = os.getpgid(process.pid)
        self.logger.warning(f"Stopping process group PGID={pgid} ({reason})")

        def _alive() -> bool:
            try:
                os.killpg(pgid, 0)  # signal 0 == check existence
                return True
            except ProcessLookupError:
                return False

        # 1) Try SIGINT (Ctrl+C) for quick graceful stop
        try:
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            return
        for _ in range(50):  # ~5s
            if process.poll() is not None or not _alive():
                break
            time.sleep(0.1)

        if process.poll() is None and _alive():
            # 2) Escalate to SIGTERM
            os.killpg(pgid, signal.SIGTERM)
            for _ in range(150):  # ~15s
                if process.poll() is not None or not _alive():
                    break
                time.sleep(0.1)

        if process.poll() is None and _alive():
            # 3) Final kill
            self.logger.warning("Escalating to SIGKILL")
            os.killpg(pgid, signal.SIGKILL)
            for _ in range(50):  # ~5s
                if process.poll() is not None or not _alive():
                    break
                time.sleep(0.1)

        # Ensure we reap it
        try:
            process.wait(timeout=2)
        except Exception:
            pass

    def _cleanup_scratch(self):
        # Clean our scratch/batch temp dirs under the output dir
        base = Path(self.meds_output_dir)
        for d in ("_scratch_tmp", "_tmp_batches"):
            p = base / d
            if p.exists():
                try:
                    shutil.rmtree(p)
                    self.logger.info(f"Cleaned {p}")
                except Exception as e:
                    self.logger.warning(f"Could not clean {p}: {e}")


    def run_preprocessing(self, num_workers: int = None, rebase: bool = True) -> Tuple[bool, Optional[Path]]:
        """Run the GenHPF MEDS preprocessing with monitoring"""
        if not self.check_prerequisites():
            return False, None

        # Calculate optimal workers
        if num_workers is None:
            num_workers = self.get_optimal_workers()
        else:
            num_workers = min(num_workers, self.get_optimal_workers())

        # Estimate processing time
        estimated_time = self.estimate_processing_time(num_workers)

        # Construct command
        command = [
            "genhpf-preprocess-meds",
            self.meds_data_dir,
            "--cohort", self.meds_labels_dir,
            "--metadata_dir", self.meds_metadata_dir,
            "--output_dir", self.meds_output_dir,
            "--workers", str(num_workers),
            "--max_event_length", str(self.max_event_length)
        ]

        if rebase:
            command.append("--rebase")

        self.logger.info("="*80)
        self.logger.info("STARTING GENHPF MEDS PREPROCESSING")
        self.logger.info("="*80)
        self.logger.info(f"Command: {' '.join(command)}")
        self.logger.info(f"Workers: {num_workers}")
        self.logger.info(f"Max event length: {self.max_event_length}")
        self.logger.info(f"Estimated time: {estimated_time}")
        self.logger.info("="*80)

        # Create output directory
        # Path(self.meds_output_dir).mkdir(parents=True, exist_ok=True)
        
        # IMPORTANT:
        # The GenHPF CLI errors if --output_dir already exists and --rebase is NOT passed.
        # So only pre-create when rebase=True (the CLI will wipe & recreate it anyway).
        if rebase:
            Path(self.meds_output_dir).mkdir(parents=True, exist_ok=True)

        # Setup log file for subprocess
        process_log = self.log_dir / f"genhpf_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._process_log_path = process_log
        self._last_process_log_size = 0
        self._seen_h5_count = self._count_output_h5()

        self.start_time = datetime.now()
        self.last_progress_time = self.start_time

        # --- redirect all temp files from /tmp to a big, known folder on the output volume ---
        scratch_dir = Path(self.meds_output_dir) / "_scratch_tmp"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # Build child environment
        child_env = os.environ.copy()
        # Generic temp vars respected by Python/tempfile and many libs
        child_env["TMPDIR"] = str(scratch_dir)
        child_env["TEMP"]  = str(scratch_dir)
        child_env["TMP"]   = str(scratch_dir)
        # Arrow spill directory (used by PyArrow/Polars)
        child_env["ARROW_TMPDIR"] = str(scratch_dir)

        # Optional: fail fast if free space looks too low (tune threshold)
        if shutil.disk_usage(scratch_dir).free < 20 * 1024**3:  # 20 GB
            self.logger.error(f"Not enough free space in {scratch_dir}; need >= 20 GB.")
            return False, process_log


        try:
            with open(process_log, "w") as log_file:
                # Launch child in its own session/group
                process = self._popen_with_own_group(
                    command,
                    stdout_handle=log_file,   # keep logging to file exactly as before
                    env=child_env,
                )
                self._write_pid_files(process.pid)

                try:
                    self.logger.info(f"Process started with PID: {process.pid} (group kill enabled)")

                    # --- wait for child to exit ---
                    rc = process.wait()
                    self.logger.info(f"Process exited with return code {rc}")

                    total_time = datetime.now() - self.start_time

                    if rc == 0:
                        self.logger.info("=" * 78)
                        self.logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
                        self.logger.info(f"Total time: {total_time}")
                        self.logger.info("=" * 78)
                        if self.verify_outputs():
                            return True, process_log
                        else:
                            self.logger.error("Output verification failed")
                            return False, process_log
                    else:
                        self.logger.error("Preprocessing failed.")
                        self.logger.error(f"Check process log for details: {process_log}")
                        self._cleanup_scratch()
                        return False, process_log

                except KeyboardInterrupt:
                    # Ctrl+C in the wrapper: stop the whole tree
                    self._terminate_process_group(process, reason="KeyboardInterrupt")
                    raise
                except Exception as e:
                    # Any wrapper error: stop the tree, log, and re-raise or return
                    self._terminate_process_group(process, reason=f"wrapper exception: {e}")
                    raise
                finally:
                    # Always clean bookkeeping; optional: clean scratch on abort
                    self._remove_pid_files()
                    # Optional: only clean scratch if nonzero exit
                    if process.poll() not in (0, None):
                        self._cleanup_scratch()

            # with open(process_log, 'w') as log_file:
            #     # Start process (own process group)
            #     # process = self._popen_with_own_group(command, log_file)
            #     process = self._popen_with_own_group(
            #         command,
            #         stdout_handle=log_file,   # keep logging EXACTLY as before
            #         env=child_env,            # <-- NEW
            #     )

            #     self.logger.info(f"Process started with PID: {process.pid}")
            #     self.logger.info(f"Process output logged to: {process_log}")

            #     # Monitor progress
            #     check_interval = 600  # 10 minutes
            #     stall_threshold = timedelta(hours=2)

            #     while process.poll() is None:
            #         time.sleep(check_interval)
            #         current_time = datetime.now()
            #         elapsed = current_time - self.start_time

            #         self.logger.info(f"Progress update - Elapsed: {elapsed}")
            #         self.monitor_resources()
            #         # Live ETA from child process tqdm
            #         if ETA_MODE == "dynamic":
            #             # Try to use total codes from Parquet; otherwise the tqdm total
            #             if not hasattr(self, "_eta_total_units"):
            #                 self._eta_total_units = self._infer_total_codes()
            #             self._dynamic_eta_update(getattr(self, "_eta_total_units", None))

            #         # Ongoing ETA
            #         ongoing_eta = self._parse_tqdm_eta_from_log(process_log)
            #         if ongoing_eta:
            #             self.logger.info(f"Ongoing ETA: {ongoing_eta}")

            #         # Progress markers (log growth or new outputs)
            #         self._update_progress_marker_if_output_advances()

            #         # Check if process is still responsive
            #         if current_time - self.last_progress_time > stall_threshold:
            #             self.logger.warning(
            #                 "No observable progress (log/output changes) for %s - process may be stuck",
            #                 stall_threshold
            #             )
            #             # Do not kill automatically; just warn. User can Ctrl-C.

            #     # Process completed
            #     return_code = process.returncode
            #     total_time = datetime.now() - self.start_time

            #     if return_code == -2:
            #         self.logger.error("Process received SIGINT (Ctrl-C or kill -2). Stopping.")
            #         self.logger.error(f"Check process log for details: {process_log}")
            #         return False, process_log

            #     if return_code == -15:
            #         self.logger.error("Process received SIGTERM (kill/termination). Stopping.")
            #         self.logger.error(f"Check process log for details: {process_log}")
            #         return False, process_log

            #     if return_code == -9:
            #         self.logger.error("Process received SIGKILL (likely OOM). "
            #                         "Try fewer --workers and/or a smaller max_event_length.")
            #         self.logger.error(f"Check process log for details: {process_log}")
            #         return False, process_log

            #     if return_code == 0:
            #         self.logger.info("="*80)
            #         self.logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
            #         self.logger.info(f"Total time: {total_time}")
            #         self.logger.info("="*80)

            #         # Verify outputs
            #         if self.verify_outputs():
            #             self.logger.info("Output verification passed")
            #             return True, process_log
            #         else:
            #             self.logger.error("Output verification failed")
            #             return False, process_log
            #     else:
            #         self.logger.error(f"Preprocessing failed with return code: {return_code}")
            #         self.logger.error(f"Check process log for details: {process_log}")
            #         return False, process_log

        except KeyboardInterrupt:
            self.logger.warning("Received interrupt signal - terminating process group")
            if 'process' in locals():
                self._terminate_process_group(process)
                try:
                    process.wait(timeout=30)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            return False, self._process_log_path
        except Exception as e:
            self.logger.error(f"Unexpected error during preprocessing: {e}")
            return False, self._process_log_path

    def verify_outputs(self) -> bool:
        """Verify that preprocessing outputs are valid"""
        self.logger.info("Verifying preprocessing outputs...")

        output_path = Path(self.meds_output_dir)

        # Check for expected directory structure
        expected_dirs = ['held_out', 'train', 'tuning']  # Based on your setup
        h5_files: List[Path] = []
        tsv_files: List[Path] = []

        for dir_name in expected_dirs:
            dir_path = output_path / dir_name
            if dir_path.exists():
                # Check for .h5 files
                h5_pattern = list(dir_path.glob("*.h5"))
                h5_files.extend(h5_pattern)
                self.logger.info(f"Found {len(h5_pattern)} .h5 files in {dir_name}/")

            # Check for .tsv files
            tsv_file = output_path / f"{dir_name}.tsv"
            if tsv_file.exists():
                tsv_files.append(tsv_file)
                self.logger.info(f"Found manifest file: {tsv_file}")

        # Verify .h5 file structure
        for h5_file in h5_files[:3]:  # Check first few files
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'ehr' not in f:
                        self.logger.warning(f"{h5_file.name}: Missing 'ehr' group")
                    else:
                        ehr_group = f['ehr']
                        self.logger.info(f"{h5_file.name}: Contains {len(ehr_group.keys())} patient cohorts")
            except Exception as e:
                self.logger.error(f"Error reading {h5_file.name}: {e}")

        success = len(h5_files) > 0 and len(tsv_files) > 0

        if success:
            self.logger.info(f"Verification complete: {len(h5_files)} .h5 files, {len(tsv_files)} .tsv files")
        else:
            self.logger.error("Verification failed: Missing expected output files")

        return success


def main():
    """Main execution function"""

    # Configuration - UPDATE THESE PATHS FOR YOUR SETUP
    # MEDS_DATA_DIR = "/opt/data/workingdir/ckravit/MEDS/MEDS220/DATA/meds_output/data/"
    # MEDS_LABELS_DIR = "/opt/data/workingdir/ckravit/MEDS/MEDS220/ACES/testhort/results/"
    # MEDS_METADATA_DIR = "/opt/data/workingdir/ckravit/MEDS/MEDS220/DATA/meds_output/metadata/"
    # MEDS_OUTPUT_DIR = "/opt/data/workingdir/ckravit/MEDS/MEDS220/genHPF/output/"
    src_dir = '/opt/data/workingdir/ckravit/MEDS/MEDS220/genHPF/'
    MEDS_DATA_DIR = '/opt/data/workingdir/ckravit/MEDS/MEDS220/DATA/MEDS_output/data/'

    # Leave empty ("") when you want to process the whole directory
    MEDS_INDIV_PARQUET = '/opt/data/workingdir/ckravit/MEDS/MEDS220/DATA/MEDS_output/data/train/0.parquet'

    # If doing a specific MEDS_INDIV_PARQUET file, make sure you point to the right subfolder
    MEDS_LABELS_DIR = '/opt/data/workingdir/ckravit/MEDS/MEDS220/ACES/testhort/results/'
    MEDS_LABELS_DIR = '/opt/data/workingdir/ckravit/MEDS/MEDS220/ACES/testhort/results/train/'


    MEDS_METADATA_DIR = '/opt/data/workingdir/ckravit/MEDS/MEDS220/DATA/MEDS_output/metadata/'
    
    # MEDS_OUTPUT_DIR = f'{src_dir}output/'
    # Base output directory (do NOT point this at '.../train/')
    BASE_OUTPUT_DIR = '/opt/data/commonfilesharePHI/MEDS_shared/ckravit/genhpf/output'

    # Decide if we're running a single file or a directory
    SINGLE_FILE = bool(MEDS_INDIV_PARQUET) and Path(MEDS_INDIV_PARQUET).exists()

    if SINGLE_FILE:
        # Single-file mode: do NOT use --rebase. Always a fresh run dir.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        MEDS_OUTPUT_DIR = str(Path(BASE_OUTPUT_DIR) / f"run_{ts}")
    else:
        # Directory mode: use the base; run_preprocessing will pass --rebase=True
        MEDS_OUTPUT_DIR = BASE_OUTPUT_DIR


    # MEDS_OUTPUT_DIR = '/opt/data/commonfilesharePHI/MEDS_shared/ckravit/genhpf/output/train/'
    # # --- Auto-switch output dir if running a SINGLE file (no rebase) and the dir already exists ---
    # # Decide if we're running a single file or a directory
    # SINGLE_FILE = bool(MEDS_INDIV_PARQUET) and Path(MEDS_INDIV_PARQUET).exists()

    # # If SINGLE FILE and output_dir already exists (and we won't rebase), pick a fresh output dir to avoid the CLI "exists" error
    # if SINGLE_FILE:
    #     out_path = Path(MEDS_OUTPUT_DIR)
    #     if out_path.exists():
    #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         MEDS_OUTPUT_DIR = str(out_path.parent / f"{out_path.name}_run_{ts}")


    # Processing parameters
    MAX_EVENT_LENGTH = 256  # Increased from default 128 to handle longer events
    MAX_WORKERS = 2  # Let system auto-calculate (set to None), or set specific number

    # Choose the data argument: single file if provided and exists; otherwise the directory
    DATA_ARG = MEDS_INDIV_PARQUET if (MEDS_INDIV_PARQUET and Path(MEDS_INDIV_PARQUET).exists()) else MEDS_DATA_DIR

    # Initialize preprocessor
    preprocessor = GenHPFPreprocessor(
        meds_data_dir=str(DATA_ARG),
        meds_labels_dir=MEDS_LABELS_DIR,
        meds_metadata_dir=MEDS_METADATA_DIR,
        meds_output_dir=MEDS_OUTPUT_DIR,
        max_event_length=MAX_EVENT_LENGTH
    )
    log = preprocessor.logger

    if str(DATA_ARG).endswith(".parquet") or str(DATA_ARG).endswith(".csv"):
        log.info(f"[Runner] Mode: SINGLE FILE → {DATA_ARG}")
    else:
        log.info(f"[Runner] Mode: DIRECTORY   → {DATA_ARG}")
    log.info(f"[Runner] Output dir: {MEDS_OUTPUT_DIR}")


    # # Run preprocessing
    # success, process_log_path = preprocessor.run_preprocessing(
    #     num_workers=MAX_WORKERS,
    #     rebase=True
    # )

    # If running a single Parquet, do NOT rebase the whole output dir.
    # For directory-wide runs, you can keep rebase=True.
    REBASE = not SINGLE_FILE   # directory run → True; single file → False
    log.info(f"[Runner] Rebase: {REBASE}")
    success, process_log_path = preprocessor.run_preprocessing(
        num_workers=MAX_WORKERS,
        rebase=REBASE
    )
    
    if success:
        # print("\n" + "="*80)
        log.info("="*80)
        log.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
        log.info("="*80)
        log.info(f"Output directory: {MEDS_OUTPUT_DIR}")
        log.info("\nNext steps:")
        log.info("1. Verify your .h5 and .tsv files")
        log.info("2. Run training with:")
        log.info(f"   genhpf-train dataset.data={MEDS_OUTPUT_DIR} \\")
        log.info(f"     model.encoder_max_seq_len=256/384/512 \\")
        log.info("     --config-dir ${GENHPF_DIR}/examples/train/genhpf \\")
        log.info("     --config-name meds_hierarchical_scr")
        log.info("\n3. For testing, use:")
        log.info(f"   genhpf-test dataset.data={MEDS_OUTPUT_DIR} \\")
        log.info("     model.encoder_max_seq_len=256/384/512 \\")
        log.info("     checkpoint.load_checkpoint=/path/to/checkpoint.pt")
        sys.exit(0)
    else:
        # print("\n" + "="*80)
        log.info("="*80)
        log.info("PREPROCESSING FAILED!")
        log.info("="*80)
        log.error("Check the log files for detailed error information.")
        if process_log_path:
            log.error(f"Process log: {process_log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
