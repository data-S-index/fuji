"""Process Fuji scores by fetching jobs from API and calling FUJI directly via Python."""

import os
import random
import time
import threading
import argparse
import signal
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import traceback

import requests

from fuji_server.controllers.fair_object_controller import evaluate_fairness

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

DOMAIN = "https://scholardata.io"


# API endpoints for fetching jobs and posting results
JOBS_API_URL = f"{DOMAIN}/api/fuji/jobs"
PRIORITY_JOBS_API_URL = f"{DOMAIN}/api/fuji/jobs/priority"
RESULTS_API_URL = f"{DOMAIN}/api/fuji/jobs/results"


def random_sleep(min_seconds: float = 30.0, max_seconds: float = 60.0) -> None:
    """
    Sleep for a random duration between min_seconds and max_seconds.
    Checks for shutdown event periodically during sleep.

    Args:
        min_seconds: Minimum sleep duration
        max_seconds: Maximum sleep duration
    """
    sleep_time = random.uniform(min_seconds, max_seconds)
    # Sleep in small chunks to check for shutdown
    chunk_size = 1.0  # Check every second
    elapsed = 0.0
    while elapsed < sleep_time and not shutdown_event.is_set():
        remaining = min(chunk_size, sleep_time - elapsed)
        if shutdown_event.wait(timeout=remaining):
            break
        elapsed += remaining


# Global shutdown event for graceful thread termination
shutdown_event = threading.Event()

# Global machine name for identifying which machine processed the results
MACHINE_NAME = "Unknown"


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)."""
    if not shutdown_event.is_set():
        print("\n\n⚠️  Shutdown signal received. Shutting down threads...")
        shutdown_event.set()


def get_fuji_score_and_date(
    result,
) -> Tuple[Optional[float], Optional[datetime]]:
    """
    Extract score and evaluation date from FUJI result object.

    Args:
        result: FAIRResults object from evaluate_fairness()

    Returns:
        Tuple of (score, evaluation_date)
        Score is 0.0-100.0, or None if extraction fails
        Evaluation date is a datetime object, or None if extraction fails
    """
    print("  🔍 Extracting score and date from FUJI result...")
    if not result:
        print(f"  ⚠️  Invalid result: {type(result)}")
        return None, None

    # Extract FAIR score from result.summary["score_percent"]["FAIR"]
    score = None
    try:
        summary = result.summary
        print(
            f"  📋 Summary keys: {list(summary.keys()) if isinstance(summary, dict) else 'not a dict'}"
        )
        score_percent = summary.get("score_percent", {})
        print(f"  📊 Score percent: {score_percent}")
        score = score_percent.get("FAIR")
        print(f"  🎯 Raw FAIR score: {score}")
        if score is not None:
            score = float(score)
            print(f"  ✅ Extracted score: {score}")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        print(f"  ⚠️  Error extracting score: {e}")
        score = None

    # Extract evaluation date from result.end_timestamp
    evaluation_date = None
    try:
        end_timestamp = result.end_timestamp
        print(f"  📅 Raw end_timestamp: {end_timestamp} (type: {type(end_timestamp)})")
        if end_timestamp:
            # Parse ISO format timestamp
            if isinstance(end_timestamp, str):
                # Handle various timestamp formats
                if end_timestamp.endswith("Z"):
                    end_timestamp = f"{end_timestamp[:-1]}+00:00"
                evaluation_date = datetime.fromisoformat(
                    end_timestamp.replace("Z", "+00:00")
                )
                print(f"  ✅ Parsed evaluation date: {evaluation_date}")
            elif isinstance(end_timestamp, datetime):
                evaluation_date = end_timestamp
                print(f"  ✅ Using datetime object: {evaluation_date}")
            elif isinstance(end_timestamp, (int, float)):
                # Handle Unix timestamp
                evaluation_date = datetime.fromtimestamp(end_timestamp)
                print(f"  ✅ Parsed evaluation date from timestamp: {evaluation_date}")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        print(f"  ⚠️  Error extracting evaluation date: {e}")
        evaluation_date = None

    return score, evaluation_date


def fetch_jobs_from_api() -> List[Dict[str, Any]]:
    """
    Fetch jobs from the API endpoints (priority first, then regular).
    Uses exponential backoff retry logic for API errors.

    Returns:
        List of job dictionaries with 'id' and 'identifier' keys
    """
    print("  🔄 Starting to fetch jobs from API...")
    all_jobs = []

    # Check for shutdown before starting
    if shutdown_event.is_set():
        print("  ⚠️  Shutdown requested, skipping job fetch")
        return all_jobs

    # Fetch priority jobs first with exponential backoff retry
    print(f"  📥 Step 1: Fetching priority jobs from {PRIORITY_JOBS_API_URL}...")
    priority_jobs = []
    for attempt in range(MAX_RETRIES):
        if shutdown_event.is_set():
            print("  ⚠️  Shutdown requested during priority jobs fetch")
            break

        try:
            print(f"  🔄 Priority jobs attempt {attempt + 1}/{MAX_RETRIES}...")
            response = requests.get(PRIORITY_JOBS_API_URL, timeout=30.0)
            response.raise_for_status()
            priority_jobs = response.json()
            print(
                f"  ✅ Priority jobs response: {len(priority_jobs) if isinstance(priority_jobs, list) else 'not a list'}"
            )
            if isinstance(priority_jobs, list):
                all_jobs.extend(priority_jobs)
                print(f"  📋 Added {len(priority_jobs)} priority jobs")
            break  # Success, exit retry loop
        except KeyboardInterrupt:
            # Re-raise to allow proper shutdown handling
            raise
        except Exception as e:
            print(
                f"  ⚠️  Error fetching priority jobs (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES - 1:
                backoff_time = RETRY_DELAY * (
                    2**attempt
                )  # Exponential backoff: 2s, 4s, 8s
                print(
                    f"  ⏳ Waiting {backoff_time}s before retry (exponential backoff)..."
                )
                if shutdown_event.wait(timeout=backoff_time):
                    print("  ⚠️  Shutdown requested during backoff")
                    break
            else:
                print(
                    f"  ❌ Failed to fetch priority jobs after {MAX_RETRIES} attempts"
                )

    # Check for shutdown before fetching regular jobs
    if shutdown_event.is_set():
        print("  ⚠️  Shutdown requested, skipping regular jobs fetch")
        return all_jobs

    # Fetch regular jobs with exponential backoff retry
    print(f"  📥 Step 2: Fetching regular jobs from {JOBS_API_URL}...")
    regular_jobs = []
    for attempt in range(MAX_RETRIES):
        if shutdown_event.is_set():
            print("  ⚠️  Shutdown requested during regular jobs fetch")
            break

        try:
            print(f"  🔄 Regular jobs attempt {attempt + 1}/{MAX_RETRIES}...")
            response = requests.get(JOBS_API_URL, timeout=120.0)
            response.raise_for_status()
            regular_jobs = response.json()
            print(
                f"  ✅ Regular jobs response: {len(regular_jobs) if isinstance(regular_jobs, list) else 'not a list'}"
            )
            if isinstance(regular_jobs, list):
                all_jobs.extend(regular_jobs)
                print(f"  📋 Added {len(regular_jobs)} regular jobs")
            break  # Success, exit retry loop
        except KeyboardInterrupt:
            # Re-raise to allow proper shutdown handling
            raise
        except Exception as e:
            print(
                f"  ⚠️  Error fetching regular jobs (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES - 1:
                backoff_time = RETRY_DELAY * (
                    2**attempt
                )  # Exponential backoff: 2s, 4s, 8s
                print(
                    f"  ⏳ Waiting {backoff_time}s before retry (exponential backoff)..."
                )
                if shutdown_event.wait(timeout=backoff_time):
                    print("  ⚠️  Shutdown requested during backoff")
                    break
            else:
                print(f"  ❌ Failed to fetch regular jobs after {MAX_RETRIES} attempts")

    # remove any jobs that don't have an identifierType of 'doi'
    print("  🔍 Step 3: Filtering jobs to only include 'doi' identifierType...")
    jobs_before_filter = len(all_jobs)
    all_jobs = [job for job in all_jobs if job.get("identifierType") == "doi"]
    jobs_filtered = jobs_before_filter - len(all_jobs)
    if jobs_filtered > 0:
        print(f"  🗑️  Filtered out {jobs_filtered} jobs (non-DOI identifierType)")

    print(f"  📊 Step 4: Total jobs fetched and filtered: {len(all_jobs)}")
    return all_jobs


def score_job(
    job: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Score a single job by calling FUJI directly via Python.

    Args:
        job: Dictionary with 'id' and 'identifier' keys

    Returns:
        Result dictionary matching the schema, or None if failed
    """
    job_id = job.get("id")
    identifier = job.get("identifier")

    print(f"  🎯 Starting to score job {job_id} with identifier: {identifier}")

    if not job_id or not identifier:
        print(f"  ⚠️  Invalid job: missing id or identifier. Job: {job}")
        return None

    # Retry logic for evaluation calls
    for attempt in range(MAX_RETRIES):
        # Check for shutdown before each attempt
        if shutdown_event.is_set():
            print(f"  ⚠️  Shutdown requested, aborting job {job_id}")
            return None

        try:
            print(f"  🔄 Attempt {attempt + 1}/{MAX_RETRIES} for job {job_id}")
            # Sleep before making FUJI evaluation call
            print("  ⏳ Random sleep before FUJI evaluation call...")
            random_sleep()
            print("  ✅ Sleep completed")

            # Check for shutdown after sleep
            if shutdown_event.is_set():
                print(f"  ⚠️  Shutdown requested, aborting job {job_id}")
                return None

            print(
                f"  📤 Step 1/3: Calling FUJI evaluate_fairness() for identifier: {identifier}"
            )
            # Call FUJI directly via Python function
            result = evaluate_fairness(
                object_identifier=identifier,
                test_debug=True,  # Disable debug for production
                use_datacite=True,
                use_github=False,
                metric_version="metrics_v0.8",
            )

            print(f"  ✅ Step 1/3: FUJI evaluation completed for job {job_id}")
            print("  📊 Step 2/3: Extracting score and metadata from result...")

            # Extract score and evaluation date from result
            score, evaluation_date = get_fuji_score_and_date(result)
            print(
                f"  ✅ Step 2/3: Score extraction completed (score: {score}, date: {evaluation_date})"
            )

            # Only return result if we successfully extracted a score
            # Skip jobs where score extraction failed
            if score is None:
                print(f"  ⚠️  Failed to extract score for job {job_id}, skipping")
                return None

            # Use current time if evaluation_date is not available
            if evaluation_date is None:
                evaluation_date = datetime.now()
                print(f"  ⏰ Using current time as evaluation date: {evaluation_date}")

            # Extract metric version and software version from FUJI result
            print("  📦 Step 3/3: Extracting metric and software versions...")
            metric_version = result.metric_version or "metrics_v0.8"
            software_version = result.software_version or "unknown"
            print(
                f"  📦 Metric version: {metric_version}, Software version: {software_version}"
            )

            # Return result in the specified schema format
            result_dict = {
                "datasetId": job_id,
                "score": float(score),
                "evaluationDate": evaluation_date.isoformat(),
                "metricVersion": str(metric_version),
                "softwareVersion": str(software_version),
            }
            print(f"  ✅ Step 3/3: Successfully scored job {job_id}: {result_dict}")
            return result_dict

        except Exception as e:
            print(traceback.format_exc())
            print(f"  ⚠️  Error for job {job_id} (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                backoff_time = RETRY_DELAY * (attempt + 1)
                print(f"  ⏳ Waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)  # Exponential backoff
                continue
            print(f"  ❌ Error for job {job_id} after {MAX_RETRIES} attempts: {str(e)}")
            return None

    print(f"  ❌ Failed to score job {job_id} after all retries")
    return None


def is_valid_result(result: Dict[str, Any]) -> bool:
    """
    Validate that a result has all required fields and valid data.

    Args:
        result: Result dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    if not result or not isinstance(result, dict):
        return False

    required_fields = [
        "datasetId",
        "score",
        "evaluationDate",
        "metricVersion",
        "softwareVersion",
    ]
    for field in required_fields:
        if field not in result:
            return False

    # Validate data types
    if not isinstance(result.get("datasetId"), (int, float)):
        return False
    if not isinstance(result.get("score"), (int, float)):
        return False
    if not isinstance(result.get("evaluationDate"), str):
        return False
    if not isinstance(result.get("metricVersion"), str):
        return False
    if not isinstance(result.get("softwareVersion"), str):
        return False

    return True


def post_results_to_api(results: List[Dict[str, Any]]) -> bool:
    """
    POST results to the API endpoint.

    Args:
        results: List of result dictionaries

    Returns:
        True if successful, False otherwise
    """
    print("  🔄 Starting to post results to API...")
    if not results:
        print("  ℹ️  No results to post")
        return True

    # Check for shutdown before posting
    if shutdown_event.is_set():
        print("  ⚠️  Shutdown requested, skipping result posting")
        return False

    payload = {"results": results, "machineName": MACHINE_NAME}
    print(
        f"  📤 Step 1/2: Preparing to post {len(results)} results to {RESULTS_API_URL}"
    )
    print(
        f"  📋 Results summary: {[{'datasetId': r.get('datasetId'), 'score': r.get('score')} for r in results[:5]]}"
    )
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more results")

    try:
        print("  📤 Step 2/2: Sending POST request to API...")
        response = requests.post(
            RESULTS_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )
        print(f"  📥 Response status: {response.status_code}")
        print(f"  📄 Response text: {response.text[:200]}")
        response.raise_for_status()
        print(f"  ✅ Step 2/2: Successfully posted {len(results)} results")
        return True
    except KeyboardInterrupt:
        # Re-raise to allow proper shutdown handling
        raise
    except requests.RequestException as e:
        print(f"  ⚠️  Request error posting results: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  📄 Error response: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"  ⚠️  Unexpected error posting results: {str(e)}")
        return False


def worker_thread(thread_id: int) -> None:
    """
    Worker thread function that continuously fetches jobs, processes them, and posts results.

    Args:
        thread_id: Unique identifier for this thread
    """
    print(f"🧵 Thread {thread_id} starting...")
    batch_count = 0

    while not shutdown_event.is_set():
        try:
            batch_count += 1
            print(
                f"\n🧵 Thread {thread_id}: ========== Starting batch #{batch_count} =========="
            )

            # Check for shutdown before starting
            if shutdown_event.is_set():
                print(f"🧵 Thread {thread_id}: ⚠️  Shutdown detected before batch start")
                break

            # Fetch jobs from API
            print(f"🧵 Thread {thread_id}: 📥 Step 1/4 - Fetching jobs from API...")
            jobs = fetch_jobs_from_api()
            print(f"🧵 Thread {thread_id}: ✅ Step 1/4 - Completed fetching jobs")

            # Check for shutdown after fetching
            if shutdown_event.is_set():
                print(
                    f"🧵 Thread {thread_id}: ⚠️  Shutdown detected after fetching jobs"
                )
                break

            if not jobs:
                print(
                    f"🧵 Thread {thread_id}: ℹ️  No jobs found, waiting 10s before next fetch..."
                )
                # Check shutdown event during wait
                if shutdown_event.wait(timeout=10):
                    print(f"🧵 Thread {thread_id}: ⚠️  Shutdown detected during wait")
                    break
                print(f"🧵 Thread {thread_id}: 🔄 Continuing to next batch...")
                continue

            # Sleep before processing jobs
            print(
                f"🧵 Thread {thread_id}: ⏳ Step 2/4 - Random sleep before processing jobs..."
            )
            random_sleep()
            print(f"🧵 Thread {thread_id}: ✅ Step 2/4 - Sleep completed")

            # Check for shutdown after sleep
            if shutdown_event.is_set():
                print(
                    f"🧵 Thread {thread_id}: ⚠️  Shutdown requested after sleep, stopping..."
                )
                break

            print(
                f"🧵 Thread {thread_id}: 📊 Step 3/4 - Processing {len(jobs):,} jobs..."
            )

            # Process all jobs
            results = []
            print(f"🧵 Thread {thread_id}: 🔄 Starting to process {len(jobs)} jobs...")

            for idx, job in enumerate(jobs, 1):
                # Check for shutdown signal
                if shutdown_event.is_set():
                    print(
                        f"🧵 Thread {thread_id}: ⚠️  Shutdown requested during job processing, stopping..."
                    )
                    break

                print(
                    f"🧵 Thread {thread_id}: 📝 Processing job {idx}/{len(jobs)} (ID: {job.get('id')}, Identifier: {job.get('identifier')})"
                )
                result = score_job(job)
                if result is not None:
                    if is_valid_result(result):
                        print(
                            f"🧵 Thread {thread_id}: ✅ Valid result for job {job.get('id')} (score: {result.get('score')})"
                        )
                        results.append(result)
                    else:
                        print(
                            f"🧵 Thread {thread_id}: ⚠️  Invalid result for job {job.get('id')}: {result}"
                        )
                else:
                    print(
                        f"🧵 Thread {thread_id}: ⚠️  No result returned for job {job.get('id')}"
                    )
                print(
                    f"🧵 Thread {thread_id}: 📊 Progress: {idx}/{len(jobs)} jobs processed, {len(results)} valid results so far"
                )

            print(
                f"🧵 Thread {thread_id}: ✅ Step 3/4 - Completed processing all {len(jobs)} jobs, got {len(results)} valid results"
            )

            # Post results to API after processing all jobs
            print(
                f"🧵 Thread {thread_id}: 📤 Step 4/4 - Posting {len(results)} results to API..."
            )
            if results:
                success = post_results_to_api(results)
                if success:
                    print(
                        f"🧵 Thread {thread_id}: ✅ Step 4/4 - Successfully posted {len(results)} results"
                    )
                else:
                    print(
                        f"🧵 Thread {thread_id}: ❌ Step 4/4 - Failed to post {len(results)} results"
                    )
            else:
                print(
                    f"🧵 Thread {thread_id}: ℹ️  Step 4/4 - No results to post (skipping API call)"
                )

            print(
                f"🧵 Thread {thread_id}: ✅ Completed batch #{batch_count} (processed {len(jobs)} jobs, got {len(results)} results)"
            )

            # Sleep before fetching next batch of jobs
            print(f"🧵 Thread {thread_id}: ⏳ Random sleep before next batch...")
            random_sleep()
            print(f"🧵 Thread {thread_id}: ✅ Sleep completed, ready for next batch")

            # Check for shutdown after sleep
            if shutdown_event.is_set():
                print(
                    f"🧵 Thread {thread_id}: ⚠️  Shutdown requested after sleep, stopping..."
                )
                break

        except Exception as e:
            if shutdown_event.is_set():
                print(
                    f"🧵 Thread {thread_id}: ⚠️  Shutdown requested during error handling"
                )
                break
            print(f"🧵 Thread {thread_id}: ❌ Error in batch #{batch_count}: {e}")
            print(f"🧵 Thread {thread_id}: 📋 Error details: {traceback.format_exc()}")
            print(f"🧵 Thread {thread_id}: ⏳ Waiting 5s before retrying next batch...")
            # Check shutdown event during wait
            if shutdown_event.wait(timeout=5):
                print(f"🧵 Thread {thread_id}: ⚠️  Shutdown detected during wait")
                break
            print(f"🧵 Thread {thread_id}: 🔄 Retrying next batch...")
            continue

    print(f"🧵 Thread {thread_id}: 🛑 Stopped")


def main(instance_count: int) -> None:
    """
    Main function to create and start worker threads that run continuously.

    Args:
        instance_count: Number of worker threads to run in parallel
    """
    print("🚀 Starting Fuji score processing...")
    print(f"🧵 Creating {instance_count} worker thread(s)...")
    print(f"🌐 Domain: {DOMAIN}")
    print(f"📥 Jobs API: {JOBS_API_URL}")
    print(f"📥 Priority Jobs API: {PRIORITY_JOBS_API_URL}")
    print(f"📤 Results API: {RESULTS_API_URL}")
    print("💡 Using FUJI Python library directly (no HTTP API calls)")
    print("💡 Program will run continuously. Press Ctrl+C to stop.\n")

    threads = []

    # Create and start threads
    for i in range(instance_count):
        print(f"🔧 Creating thread {i + 1}")
        thread = threading.Thread(
            target=worker_thread, args=(i + 1,), name=f"Worker-{i + 1}", daemon=False
        )
        threads.append(thread)
        thread.start()
        print(f"✅ Thread {i + 1} started")

    # Wait for all threads to complete (they run indefinitely until interrupted)
    print(f"\n🔄 All {len(threads)} threads started. Waiting for completion...\n")

    # Poll for shutdown instead of blocking indefinitely
    try:
        while not shutdown_event.is_set():
            # Check if all threads are still alive
            alive_threads = [t for t in threads if t.is_alive()]
            if not alive_threads:
                break

            # Wait a short time and check again
            shutdown_event.wait(timeout=1.0)

            # If shutdown was requested, break immediately
            if shutdown_event.is_set():
                break
    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt detected. Shutting down threads...")
        shutdown_event.set()

    # Now wait for threads to finish gracefully
    if shutdown_event.is_set():
        print("🛑 Shutdown signal sent to all threads")
        # Wait for threads to finish current iteration with timeout
        for thread in threads:
            if thread.is_alive():
                print(
                    f"⏳ Waiting for {thread.name} to finish gracefully (timeout: 30s)..."
                )
                thread.join(timeout=30)  # Give threads more time to finish gracefully
                if thread.is_alive():
                    print(f"⚠️  {thread.name} did not finish within timeout")

    print("\n✅ All threads stopped!")


if __name__ == "__main__":
    # Register signal handlers early, before any threads are created
    signal.signal(signal.SIGINT, signal_handler)
    # SIGTERM is not available on Windows
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description=(
            "Process Fuji scores by fetching jobs from API "
            "and calling FUJI directly via Python."
        )
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("THREADS", "1")),
        help=(
            "Number of worker threads to run in parallel "
            "(default: 1, or THREADS env var)"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Machine name to identify which machine processed the results",
    )

    args = parser.parse_args()

    INSTANCE_COUNT = args.threads
    if MACHINE_NAME := args.name:
        print(f"🏷️  Machine name: {MACHINE_NAME}")
    print(f"🚀 Starting Fuji score processing with {INSTANCE_COUNT} threads...")

    try:
        main(INSTANCE_COUNT)
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        shutdown_event.set()  # Ensure shutdown is set
        exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        shutdown_event.set()  # Ensure shutdown is set on error
        exit(1)
