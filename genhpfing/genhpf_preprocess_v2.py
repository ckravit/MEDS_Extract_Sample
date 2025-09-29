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
        """Estimate processing time based on data size and workers"""
        try:
            # Count data files
            data_files: List[str] = []
            data_path = Path(self.meds_data_dir)
            if data_path.is_dir():
                data_files.extend(glob.glob(str(data_path / "**/*.csv"), recursive=True))
                data_files.extend(glob.glob(str(data_path / "**/*.parquet"), recursive=True))
            else:
                data_files = [str(data_path)]

            total_size_gb = sum(Path(f).stat().st_size for f in data_files) / (1024**3)

            # Heuristic: ~1 GB per hour per worker
            hours_per_gb = 1.0
            estimated_hours = max(0.5, (total_size_gb * hours_per_gb) / max(1, num_workers))

            self.logger.info(f"Data size: {total_size_gb:.1f}GB across {len(data_files)} files")
            self.logger.info(f"Estimated processing time: {estimated_hours:.1f} hours with {num_workers} workers")

            return f"{estimated_hours:.1f} hours"
        except Exception as e:
            self.logger.warning(f"Could not estimate processing time: {e}")
            return "unknown"

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

    def _popen_with_own_group(self, command, stdout_handle):
        """Start subprocess in its own process group for group termination"""
        if os.name == "nt":
            # Windows: CREATE_NEW_PROCESS_GROUP
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            return subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=CREATE_NEW_PROCESS_GROUP
            )
        else:
            # POSIX: new session (setsid)
            return subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid
            )

    def _terminate_process_group(self, process: subprocess.Popen):
        """Terminate the entire child process group"""
        try:
            if os.name == "nt":
                # send CTRL-BREAK to the process group; fall back to terminate
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                except Exception:
                    process.terminate()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

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

        try:
            with open(process_log, 'w') as log_file:
                # Start process (own process group)
                process = self._popen_with_own_group(command, log_file)

                self.logger.info(f"Process started with PID: {process.pid}")
                self.logger.info(f"Process output logged to: {process_log}")

                # Monitor progress
                check_interval = 600  # 10 minutes
                stall_threshold = timedelta(hours=2)

                while process.poll() is None:
                    time.sleep(check_interval)
                    current_time = datetime.now()
                    elapsed = current_time - self.start_time

                    self.logger.info(f"Progress update - Elapsed: {elapsed}")
                    self.monitor_resources()

                    # Progress markers (log growth or new outputs)
                    self._update_progress_marker_if_output_advances()

                    # Check if process is still responsive
                    if current_time - self.last_progress_time > stall_threshold:
                        self.logger.warning(
                            "No observable progress (log/output changes) for %s - process may be stuck",
                            stall_threshold
                        )
                        # Do not kill automatically; just warn. User can Ctrl-C.

                # Process completed
                return_code = process.returncode
                total_time = datetime.now() - self.start_time

                if return_code == -2:
                    self.logger.error("Process received SIGINT (Ctrl-C or kill -2). Stopping.")
                    self.logger.error(f"Check process log for details: {process_log}")
                    return False, process_log

                if return_code == -15:
                    self.logger.error("Process received SIGTERM (kill/termination). Stopping.")
                    self.logger.error(f"Check process log for details: {process_log}")
                    return False, process_log

                if return_code == -9:
                    self.logger.error("Process received SIGKILL (likely OOM). "
                                    "Try fewer --workers and/or a smaller max_event_length.")
                    self.logger.error(f"Check process log for details: {process_log}")
                    return False, process_log

                if return_code == 0:
                    self.logger.info("="*80)
                    self.logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
                    self.logger.info(f"Total time: {total_time}")
                    self.logger.info("="*80)

                    # Verify outputs
                    if self.verify_outputs():
                        self.logger.info("Output verification passed")
                        return True, process_log
                    else:
                        self.logger.error("Output verification failed")
                        return False, process_log
                else:
                    self.logger.error(f"Preprocessing failed with return code: {return_code}")
                    self.logger.error(f"Check process log for details: {process_log}")
                    return False, process_log

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
    MEDS_OUTPUT_DIR = '/opt/data/commonfilesharePHI/MEDS_shared/ckravit/genhpf/output/pretrain/'

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

    # # Run preprocessing
    # success, process_log_path = preprocessor.run_preprocessing(
    #     num_workers=MAX_WORKERS,
    #     rebase=True
    # )

    # If running a single Parquet, do NOT rebase the whole output dir.
    # For directory-wide runs, you can keep rebase=True.
    REBASE = not (MEDS_INDIV_PARQUET and Path(MEDS_INDIV_PARQUET).exists())

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
