"""Process Fuji scores by fetching jobs from API and calling FUJI directly via Python."""

import os
import random
import time
import threading
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import traceback

import requests

from fuji_server.controllers.fair_object_controller import evaluate_fairness

# Number of worker threads to run in parallel
INSTANCE_COUNT = int(os.getenv("INSTANCE_COUNT", "30"))

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

    Args:
        min_seconds: Minimum sleep duration
        max_seconds: Maximum sleep duration
    """
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)


# Global shutdown event for graceful thread termination
shutdown_event = threading.Event()


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

    Returns:
        List of job dictionaries with 'id' and 'identifier' keys
    """
    all_jobs = []

    # Fetch priority jobs first
    print(f"  📥 Fetching priority jobs from {PRIORITY_JOBS_API_URL}...")
    try:
        response = requests.get(PRIORITY_JOBS_API_URL, timeout=30.0)
        response.raise_for_status()
        priority_jobs = response.json()
        print(
            f"  ✅ Priority jobs response: {len(priority_jobs) if isinstance(priority_jobs, list) else 'not a list'}"
        )
        if isinstance(priority_jobs, list):
            all_jobs.extend(priority_jobs)
            print(f"  📋 Added {len(priority_jobs)} priority jobs")
    except Exception as e:
        print(f"  ⚠️  Error fetching priority jobs: {e}")

    # Fetch regular jobs
    print(f"  📥 Fetching regular jobs from {JOBS_API_URL}...")
    try:
        response = requests.get(JOBS_API_URL, timeout=120.0)
        response.raise_for_status()
        regular_jobs = response.json()
        print(
            f"  ✅ Regular jobs response: {len(regular_jobs) if isinstance(regular_jobs, list) else 'not a list'}"
        )
        if isinstance(regular_jobs, list):
            all_jobs.extend(regular_jobs)
            print(f"  📋 Added {len(regular_jobs)} regular jobs")
    except Exception as e:
        print(f"  ⚠️  Error fetching regular jobs: {e}")

    # remove any jobs that don't have an identifierType of 'doi'
    all_jobs = [job for job in all_jobs if job.get("identifierType") == "doi"]

    print(f"  📊 Total jobs fetched: {len(all_jobs)}")
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

    print(f"  🎯 Scoring job {job_id} with identifier: {identifier}")

    if not job_id or not identifier:
        print(f"  ⚠️  Invalid job: missing id or identifier. Job: {job}")
        return None

    # Retry logic for evaluation calls
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  🔄 Attempt {attempt + 1}/{MAX_RETRIES} for job {job_id}")
            # Sleep before making FUJI evaluation call
            random_sleep()

            print(f"  📤 Evaluating identifier: {identifier}")
            # Call FUJI directly via Python function
            result = evaluate_fairness(
                object_identifier=identifier,
                test_debug=True,  # Disable debug for production
                use_datacite=True,
                use_github=False,
                metric_version="metrics_v0.8",
            )

            print(f"  📊 Evaluation completed for job {job_id}")

            # Extract score and evaluation date from result
            score, evaluation_date = get_fuji_score_and_date(result)

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
            print(f"  ✅ Successfully scored job {job_id}: {result_dict}")
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
    if not results:
        print("  ℹ️  No results to post")
        return True

    payload = {"results": results}
    print(f"  📤 Posting {len(results)} results to {RESULTS_API_URL}")
    print(
        f"  📋 Results summary: {[{'datasetId': r.get('datasetId'), 'score': r.get('score')} for r in results[:5]]}"
    )
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more results")

    try:
        response = requests.post(
            RESULTS_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )
        print(f"  📥 Response status: {response.status_code}")
        print(f"  📄 Response text: {response.text[:200]}")
        response.raise_for_status()
        print(f"  ✅ Successfully posted {len(results)} results")
        return True
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

    while not shutdown_event.is_set():
        try:
            # Fetch jobs from API
            print(f"🧵 Thread {thread_id}: 📥 Fetching jobs...")
            jobs = fetch_jobs_from_api()

            if not jobs:
                print(f"🧵 Thread {thread_id}: ✅ No jobs found, waiting...")
                # Check shutdown event during wait
                if shutdown_event.wait(timeout=10):
                    break
                continue

            # Sleep before processing jobs
            random_sleep()

            print(f"🧵 Thread {thread_id}: 📊 Processing {len(jobs):,} jobs...")

            # Process all jobs
            results = []
            print(f"🧵 Thread {thread_id}: 🔄 Starting to process {len(jobs)} jobs...")

            for idx, job in enumerate(jobs, 1):
                # Check for shutdown signal
                if shutdown_event.is_set():
                    print(f"🧵 Thread {thread_id}: ⚠️  Shutdown requested, stopping...")
                    break

                print(
                    f"🧵 Thread {thread_id}: 📝 Processing job {idx}/{len(jobs)}: {job}"
                )
                result = score_job(job)
                if result is not None:
                    if is_valid_result(result):
                        print(
                            f"🧵 Thread {thread_id}: ✅ Valid result for job {job.get('id')}"
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

            # Post results to API after processing all jobs
            if results:
                print(f"🧵 Thread {thread_id}: 📤 Posting {len(results)} results...")
                success = post_results_to_api(results)
                if success:
                    print(
                        f"🧵 Thread {thread_id}: ✅ Successfully posted {len(results)} results"
                    )
                else:
                    print(
                        f"🧵 Thread {thread_id}: ❌ Failed to post {len(results)} results"
                    )
            else:
                print(f"🧵 Thread {thread_id}: ℹ️  No results to post")

            print(
                f"🧵 Thread {thread_id}: ✅ Completed batch (processed {len(jobs)} jobs, got {len(results)} results)"
            )

            # Sleep before fetching next batch of jobs
            random_sleep()

        except Exception as e:
            if shutdown_event.is_set():
                print(f"🧵 Thread {thread_id}: ⚠️  Shutdown requested")
                break
            print(f"🧵 Thread {thread_id}: ❌ Error: {e}")
            print(f"🧵 Thread {thread_id}: Retrying...")
            # Check shutdown event during wait
            if shutdown_event.wait(timeout=5):
                break
            continue

    print(f"🧵 Thread {thread_id}: 🛑 Stopped")


def main() -> None:
    """Main function to create and start worker threads that run continuously."""
    print("🚀 Starting Fuji score processing...")
    print(f"🧵 Creating {INSTANCE_COUNT} worker thread(s)...")
    print(f"🌐 Domain: {DOMAIN}")
    print(f"📥 Jobs API: {JOBS_API_URL}")
    print(f"📥 Priority Jobs API: {PRIORITY_JOBS_API_URL}")
    print(f"📤 Results API: {RESULTS_API_URL}")
    print("💡 Using FUJI Python library directly (no HTTP API calls)")
    print("💡 Program will run continuously. Press Ctrl+C to stop.\n")

    threads = []

    # Create and start threads
    for i in range(INSTANCE_COUNT):
        print(f"🔧 Creating thread {i + 1}")
        thread = threading.Thread(
            target=worker_thread, args=(i + 1,), name=f"Worker-{i + 1}"
        )
        threads.append(thread)
        thread.start()
        print(f"✅ Thread {i + 1} started")

    # Wait for all threads to complete (they run indefinitely until interrupted)
    print(f"\n🔄 All {len(threads)} threads started. Waiting for completion...\n")
    try:
        for thread in threads:
            print(f"⏳ Waiting for {thread.name}...")
            thread.join()
    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt detected. Shutting down threads...")
        shutdown_event.set()  # Signal all threads to stop
        print("🛑 Shutdown signal sent to all threads")
        # Wait for threads to finish current iteration
        for thread in threads:
            print(
                f"⏳ Waiting for {thread.name} to finish gracefully (timeout: 10s)..."
            )
            thread.join(timeout=10)  # Give threads time to finish gracefully
            if thread.is_alive():
                print(f"⚠️  {thread.name} did not finish within timeout")

    print("\n✅ All threads stopped!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        exit(1)
