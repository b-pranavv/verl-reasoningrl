import json
import asyncio
import time
import numpy as np
import uuid
import random
import hashlib
from typing import Dict, Any, List, Optional, BinaryIO
from azure.identity import ManagedIdentityCredential
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool

from azure.identity import AzureCliCredential

# pip install fastapi langchain azure-identity langchain-openai python-dotenv langchainhub langchain-azure-dynamic-sessions

def token_provider(_: str = None):
    """Token provider for Azure Dynamic Sessions using Azure CLI credentials"""
    resource_scope = "https://dynamicsessions.io/.default"
    credential = AzureCliCredential()
    token = credential.get_token(resource_scope).token
    return token

class AzureDynamicSessionsClient:
    def __init__(self, session_pool_urls: List[str], token_provider_func, max_concurrent: int = 100, max_retries: int = 3):
        self.session_pool_urls = [url.rstrip('/') for url in session_pool_urls]
        self.token_provider_func = token_provider_func
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

    def get_session_pool_url_for_session(self, session_id: str) -> str:
        """Get consistent session pool URL for a given session ID using hash-based selection"""
        # Use hash of session_id to consistently map to the same pool
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        pool_index = hash_value % len(self.session_pool_urls)
        return self.session_pool_urls[pool_index]

    def get_session_pool_url(self, index: int) -> str:
        """Round-robin selection of session pool (kept for backward compatibility)"""
        return self.session_pool_urls[index % len(self.session_pool_urls)]

    def create_repl_tool(self, session_pool_url: str, session_id: str) -> SessionsPythonREPLTool:
        """Create a new REPL tool with the specified session ID"""
        return SessionsPythonREPLTool(
            name="Python Code Interpreter",
            pool_management_endpoint=session_pool_url,
            session_id=session_id,
            access_token_provider=self.token_provider_func
        )

    def _handle_file_uploads(self, repl_tool: SessionsPythonREPLTool,
                             upload_files: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Handle file upload operations"""
        file_results = []
        if upload_files:
            for file_spec in upload_files:
                try:
                    local_file = file_spec.get("local_file")
                    remote_file = file_spec.get("remote_file")

                    if not remote_file:
                        raise ValueError("remote_file must be specified for upload")

                    if isinstance(local_file, str):
                        # local_file is a file path
                        metadata = repl_tool.upload_file(
                            local_file_path=local_file,
                            remote_file_path=remote_file
                        )
                    elif hasattr(local_file, 'read'):
                        # local_file is BinaryIO data
                        metadata = repl_tool.upload_file(
                            data=local_file,
                            remote_file_path=remote_file
                        )
                    else:
                        raise ValueError("local_file must be a file path (str) or BinaryIO data")

                    file_results.append({
                        "local_file": str(local_file) if isinstance(local_file, str) else "<BinaryIO>",
                        "remote_file": remote_file,
                        "status": "success",
                        "metadata": metadata.__dict__ if hasattr(metadata, '__dict__') else str(metadata)
                    })
                except Exception as e:
                    file_results.append({
                        "local_file": str(file_spec.get("local_file", "<unknown>")),
                        "remote_file": file_spec.get("remote_file", "<unknown>"),
                        "status": "error",
                        "error": str(e)
                    })
        return file_results

    def _handle_file_downloads(self, repl_tool: SessionsPythonREPLTool,
                                 download_files: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Handle file download operations"""
        file_results = []
        if download_files:
            for file_spec in download_files:
                try:
                    local_file = file_spec.get("local_file")
                    remote_file = file_spec.get("remote_file")

                    if not remote_file:
                        raise ValueError("remote_file must be specified for download")

                    if local_file:
                        # Download to local file path
                        data = repl_tool.download_file(
                            remote_file_path=remote_file,
                            local_file_path=local_file
                        )
                        file_results.append({
                            "local_file": local_file,
                            "remote_file": remote_file,
                            "status": "success",
                            "data": None  # File saved to disk
                        })
                    else:
                        # Return BinaryIO data
                        data = repl_tool.download_file(remote_file_path=remote_file)
                        file_results.append({
                            "local_file": None,
                            "remote_file": remote_file,
                            "status": "success",
                            "data": data  # BinaryIO data
                        })
                except Exception as e:
                    file_results.append({
                        "local_file": file_spec.get("local_file"),
                        "remote_file": file_spec.get("remote_file", "<unknown>"),
                        "status": "error",
                        "error": str(e),
                        "data": None
                    })
        return file_results

    async def execute_code_with_retry(self, code: str, index: int, session_identifier: str = None,
                                    upload_files: Optional[List[Dict[str, Any]]] = None,
                                    download_files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Execute code with retry logic for 409 conflicts"""
        last_error = None

        # Determine the primary session pool based on session_identifier
        if session_identifier:
            primary_session_pool_url = self.get_session_pool_url_for_session(session_identifier)
        else:
            # Fallback to round-robin if no session identifier
            primary_session_pool_url = self.get_session_pool_url(index)

        for attempt in range(self.max_retries):
            try:
                # For the first attempt, use the consistent session pool
                # For retries, try other pools to handle conflicts
                if attempt == 0:
                    session_pool_url = primary_session_pool_url
                elif session_identifier is None:
                    # Try different session pools on retries
                    pool_index = (index + attempt) % len(self.session_pool_urls)
                    session_pool_url = self.session_pool_urls[pool_index]

                # Generate new session identifier for each retry to avoid conflicts
                if attempt > 0 or session_identifier is None:
                    session_identifier = str(uuid.uuid4())

                result = await self._execute_code_single(code, session_pool_url, session_identifier, index,
                                                       upload_files, download_files)
                return result

            except Exception as e:
                error_str = str(e)
                last_error = e

                # Backoff for 409 conflict error
                if "409" in error_str and "Conflict" in error_str:
                    print(f"Attempt {attempt + 1}: 409 Conflict for index {index}, retrying with different pool...")

                    # Exponential backoff with jitter
                    wait_time = (5 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        # If all retries failed, return error result
        raise Exception(f"Max retries exceeded for index {index}: {last_error}")s
        return {
            'index': index,
            'code': code,
            'session_identifier': session_identifier,
            'execution_time': 0,
            'success': False,
            'error': f"Max retries exceeded. Last error: {str(last_error)}",
            'output': '',
            'session_pool': primary_session_pool_url,
            'status': 'max_retries_exceeded',
            'file_operations': {"uploads": [], "downloads": []}
        }

    async def _execute_code_single(self, code: str, session_pool_url: str, session_identifier: str, index: int,
                                 upload_files: Optional[List[Dict[str, Any]]] = None,
                                 download_files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Execute code using LangChain's SessionsPythonREPLTool (single attempt)"""
        start_time = time.time()

        try:
            # Create a new REPL tool with the specific session ID
            repl_tool = self.create_repl_tool(session_pool_url, session_identifier)

            # Handle file operations
            loop = asyncio.get_event_loop()
            file_results = {}

            file_results['uploads'] = await loop.run_in_executor(None, self._handle_file_uploads, repl_tool, upload_files)

            # Running in executor to avoid blocking the event loop
            result = await loop.run_in_executor(None, repl_tool.run, code)

            file_results['downloads'] = await loop.run_in_executor(None, self._handle_file_downloads, repl_tool, download_files)

            execution_time = time.time() - start_time

            # Parse the result - SessionsPythonREPLTool returns a string
            success = True
            output = result
            error = ""

            # Check if the result contains error indicators
            if "Error" in result or "Exception" in result or "Traceback" in result:
                success = False
                error = result
                output = ""

            return {
                'index': index,
                'code': code,
                'session_identifier': session_identifier,
                'execution_time': execution_time,
                'success': success,
                'output': output,
                'error': error,
                'session_pool': session_pool_url,
                'status': 'Success' if success else 'Failed',
                'file_operations': file_results
            }
        except Exception as e:
            execution_time = time.time() - start_time
            raise e

    async def execute_code(self, code: str, index: int, session_identifier: str = None,
                          upload_files: Optional[List[Dict[str, Any]]] = None,
                          download_files: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Execute code using LangChain's SessionsPythonREPLTool with retry logic"""
        return await self.execute_code_with_retry(code, index, session_identifier, upload_files, download_files)

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of records"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}, Error: {e}")
    return records

async def execute_batch(client: AzureDynamicSessionsClient, 
                       batch: List[tuple], semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
    """Execute a batch of code snippets with concurrency control"""
    async def execute_with_semaphore(code, index, session_identifier, upload_files, download_files):
        async with semaphore:
            return await client.execute_code(code, index, session_identifier, upload_files, download_files)

    tasks = [execute_with_semaphore(code, index, session_identifier, upload_files, download_files) 
             for code, index, session_identifier, upload_files, download_files in batch]
    return await asyncio.gather(*tasks)

async def execute_sequential(client: AzureDynamicSessionsClient,
                             batch: List[tuple]) -> List[Dict[str, Any]]:
    """Execute a batch of code snippets sequentially (no concurrency)"""
    results = []
    for code, index, session_identifier, upload_files, download_files in batch:
        result = await client.execute_code(code, index, session_identifier, upload_files, download_files)
        results.append(result)
    return results    

def calculate_percentiles(execution_times: List[float]) -> Dict[str, float]:
    """Calculate percentile statistics for execution times"""
    if not execution_times:
        return {}

    percentiles = [50, 75, 90, 95, 99]
    return {f"p{p}": np.percentile(execution_times, p) for p in percentiles}

async def compute_latency_async(file_path: str, subscription_id: str, resource_group: str, 
                              mi_client_id: str, max_concurrent: int = 100, num_pools: int = 10, 
                              region: str = "westus2", execution_mode: str = "batch"):
    """Compute latency for code execution using Azure Dynamic Sessions pools with configurable execution mode"""

    # Generate session pool URLs with configurable region
    session_pool_urls = []
    for i in range(1, num_pools + 1):
        url = f"https://{region}.dynamicsessions.io/subscriptions/{subscription_id}/resourceGroups/{resource_group}/sessionPools/python-code-interpreter-pool-{i}"
        session_pool_urls.append(url)

    print(f"Using {len(session_pool_urls)} session pools in region {region}:")
    for url in session_pool_urls:
        print(f"  - {url}")

    # Create token provider function with client ID
    token_provider_func = lambda: token_provider(mi_client_id)

    # Initialize client with token provider
    client = AzureDynamicSessionsClient(session_pool_urls, token_provider_func, max_concurrent)

    # Read JSONL file
    records = read_jsonl_file(file_path)
    print(f"Loaded {len(records)} records from {file_path}")

    code_records = []
    for i, record in enumerate(records):
        if 'code' not in record:
            print(f"Record {i} does not contain 'code' field, skipping...")
            continue
        if not isinstance(record['code'], str):
            print(f"Record {i} has invalid 'code' type: {type(record['code'])}, expected str, skipping...")
            continue

        if not record['code'].strip():
            print(f"Record {i} has empty 'code' field, skipping...")
            continue

        # Store code, index, session identifier, and file operations
        session_id = record.get('session_id', str(uuid.uuid4()))
        if not session_id or not isinstance(session_id, str):
            print(f"Record {i} has invalid 'session_id', generating a new one...")
            session_id = str(uuid.uuid4())

        upload_files = record.get('upload_files', [])
        download_files = record.get('download_files', [])

        code_records.append((record['code'], i, session_id, upload_files, download_files))

    print(f"Found {len(code_records)} records with 'code' field")
    if not code_records:
        print("No valid code records found!")
        return []

    # Validate execution mode
    if execution_mode not in ["batch", "sequential"]:
        print(f"Invalid execution_mode: {execution_mode}. Using 'batch' as default.")
        execution_mode = "batch"

    print(f"Execution mode: {execution_mode}")
    if execution_mode == "batch":
        print(f"Executing {len(code_records)} code snippets with max {max_concurrent} concurrent requests...")
    else:
        print(f"Executing {len(code_records)} code snippets sequentially...")

    start_time = time.time()

    # Execute based on mode
    if execution_mode == "batch":
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        results = await execute_batch(client, code_records, semaphore)
    else:  # sequential
        results = await execute_sequential(client, code_records)

    total_time = time.time() - start_time

    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])

    # Calculate statistics
    successful_executions = [r for r in results if r['success']]
    failed_executions = [r for r in results if not r['success']]
    conflict_retries = [r for r in results if r.get('status') == 'max_retries_exceeded']

    all_execution_times = [r['execution_time'] for r in results]
    successful_execution_times = [r['execution_time'] for r in successful_executions]

    # Calculate percentiles
    all_percentiles = calculate_percentiles(all_execution_times)
    success_percentiles = calculate_percentiles(successful_execution_times)

    # Session pool usage statistics
    pool_usage = {}
    for result in results:
        pool = result.get('session_pool', 'unknown')
        pool_usage[pool] = pool_usage.get(pool, 0) + 1

    # File operation statistics
    total_uploads = sum(len(r.get('file_operations', {}).get('uploads', [])) for r in results)
    total_downloads = sum(len(r.get('file_operations', {}).get('downloads', [])) for r in results)
    successful_uploads = sum(len([u for u in r.get('file_operations', {}).get('uploads', []) if u.get('status') == 'success']) for r in results)
    successful_downloads = sum(len([d for d in r.get('file_operations', {}).get('downloads', []) if d.get('status') == 'success']) for r in results)

    # Print summary statistics
    print(f"\n--- Summary ---")
    print(f"Region: {region}")
    print(f"Execution mode: {execution_mode}")
    print(f"Total executions: {len(results)}")
    print(f"Successful: {len(successful_executions)}")
    print(f"Failed: {len(failed_executions)}")
    print(f"Conflict retries exhausted: {len(conflict_retries)}")
    print(f"Success rate: {len(successful_executions)/len(results)*100:.1f}%")
    print(f"Total wall time: {total_time:.3f}s")
    print(f"Average wall time per execution: {total_time/len(results):.3f}s")

    if successful_executions:
        avg_success_time = sum(r['execution_time'] for r in successful_executions) / len(successful_executions)
        print(f"Average individual execution time (successful): {avg_success_time:.3f}s")

    if results:
        avg_individual_time = sum(r['execution_time'] for r in results) / len(results)
        print(f"Average individual execution time (all): {avg_individual_time:.3f}s")

        if execution_mode == "batch":
            speedup = (avg_individual_time * len(results)) / total_time
            print(f"Speedup factor: {speedup:.1f}x")
        else:
            print(f"Sequential execution - no speedup calculation")

    # Print file operation statistics
    if total_uploads > 0 or total_downloads > 0:
        print(f"\n--- File Operation Statistics ---")
        print(f"Total file uploads: {total_uploads}")
        print(f"Successful uploads: {successful_uploads}")
        print(f"Total file downloads: {total_downloads}")
        print(f"Successful downloads: {successful_downloads}")
        if total_uploads > 0:
            print(f"Upload success rate: {successful_uploads/total_uploads*100:.1f}%")
        if total_downloads > 0:
            print(f"Download success rate: {successful_downloads/total_downloads*100:.1f}%")

    # Print session pool usage
    print(f"\n--- Session Pool Usage ---")
    for pool, count in pool_usage.items():
        pool_name = pool.split('/')[-1] if '/' in pool else pool
        print(f"{pool_name}: {count} executions")

    # Print percentile statistics
    print(f"\n--- Percentile Latencies (All Executions) ---")
    for percentile, value in all_percentiles.items():
        print(f"{percentile}: {value:.3f}s")

    if success_percentiles:
        print(f"\n--- Percentile Latencies (Successful Executions Only) ---")
        for percentile, value in success_percentiles.items():
            print(f"{percentile}: {value:.3f}s")

    return results

def compute_latency(file_path: str, subscription_id: str, resource_group: str, 
                   mi_client_id: str, max_concurrent: int = 100, num_pools: int = 10, 
                   region: str = "westus2", execution_mode: str = "batch"):
    """Wrapper function to run async computation"""
    return asyncio.run(compute_latency_async(file_path, subscription_id, resource_group, 
                                           mi_client_id, max_concurrent, num_pools, region, execution_mode))

async def get_results_async(code_list: str, subscription_id: str, resource_group: str, 
                              mi_client_id: str, max_concurrent: int = 100, num_pools: int = 10, 
                              region: str = "westus2", execution_mode: str = "batch"):
    """Compute latency for code execution using Azure Dynamic Sessions pools with configurable execution mode"""

    # Generate session pool URLs with configurable region
    session_pool_urls = []
    for i in range(1, num_pools + 1):
        url = f"https://{region}.dynamicsessions.io/subscriptions/{subscription_id}/resourceGroups/{resource_group}/sessionPools/python-code-interpreter-pool-{i}"
        session_pool_urls.append(url)

    print(f"Using {len(session_pool_urls)} session pools in region {region}:")
    for url in session_pool_urls:
        print(f"  - {url}")

    # Create token provider function with client ID
    token_provider_func = lambda: token_provider(mi_client_id)

    # Initialize client with token provider
    client = AzureDynamicSessionsClient(session_pool_urls, token_provider_func, max_concurrent)

    code_records = []
    for i, code in enumerate(code_list):

        if not isinstance(code, str):
            print("Code must be a string, skipping...")
            continue

        if not code.strip():
            print(f"Record {i} has empty 'code' field, skipping...")
            continue

        # Store code, index, session identifier, and file operations
        session_id = str(uuid.uuid4())

        upload_files =[]
        download_files = []

        code_records.append((code,i, session_id, upload_files, download_files))

    if not code_records:
        print("No valid code records found!")
        return []

    # Validate execution mode
    if execution_mode not in ["batch", "sequential"]:
        print(f"Invalid execution_mode: {execution_mode}. Using 'batch' as default.")
        execution_mode = "batch"

    # Execute based on mode
    if execution_mode == "batch":
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        results = await execute_batch(client, code_records, semaphore)
    else:  # sequential
        results = await execute_sequential(client, code_records)


    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])


    return results

def getResults(code_list: str, subscription_id: str, resource_group: str,
               mi_client_id: str, max_concurrent: int = 100, num_pools: int = 10, 
               region: str = "westus2", execution_mode: str = "batch"):
    """Wrapper function to run async computation"""
    return asyncio.run(get_results_async(code_list, subscription_id, resource_group, 
                                           mi_client_id, max_concurrent, num_pools, region, execution_mode))


def executeCode(code_list):
    SUBSCRIPTION_ID = "d4fe558f-6660-4fe7-99ec-ae4716b5e03f"
    RESOURCE_GROUP = "reasoning_tools"
    MI_CLIENT_ID = "b32444ac-27e2-4f36-ab71-b664f6876f00"
    MAX_CONCURRENT = 100
    NUM_POOLS = 10
    REGION = "westus2"

    EXECUTION_MODE = "sequential"

    results = getResults(code_list, SUBSCRIPTION_ID, RESOURCE_GROUP, MI_CLIENT_ID, MAX_CONCURRENT, NUM_POOLS, REGION, EXECUTION_MODE)

    return results

if __name__ == "__main__":
    # Configuration
    SUBSCRIPTION_ID = "d4fe558f-6660-4fe7-99ec-ae4716b5e03f"
    RESOURCE_GROUP = "reasoning_tools"
    MI_CLIENT_ID = "b32444ac-27e2-4f36-ab71-b664f6876f00"
    MAX_CONCURRENT = 100
    NUM_POOLS = 10
    REGION = "westus2"

    # # Batch execution example
    # JSONL_FILE = "tiny-codes-python.jsonl"
    # EXECUTION_MODE = "batch"

    # # Session Context Sequential execution example
    JSONL_FILE = "python-session-code-data-sample.jsonl"
    EXECUTION_MODE = "sequential"

    code_list = ["import pandas as pd\nimport numpy as np\n\n# Create a DataFrame with random data\nnp.random.seed(0)\ndata = pd.(np.random.randn(100, 4), columns=list('ABCD'))\n\n# Display the first few rows of the DataFrame\ndata.head()", 
                 "import requests\nfrom bs4 import BeautifulSoup\n\n# Fetch a webpage\nurl = 'https://example.com'\nresponse = requests.get(url)\n\n# Parse the HTML content\nsoup = BeautifulSoup(response.content, 'html.parser')\n\n# Extract and print the title of the page\ntitle = soup.title.string\nprint(title)"]
    results = getResults(code_list, SUBSCRIPTION_ID, RESOURCE_GROUP, MI_CLIENT_ID, MAX_CONCURRENT, NUM_POOLS, REGION, EXECUTION_MODE)

    # File Upload and Download example
    # JSONL_FILE = "python-file-code-sample.jsonl"
    # EXECUTION_MODE = "sequential"

    # Run latency computation
    # results = compute_latency(JSONL_FILE, SUBSCRIPTION_ID, RESOURCE_GROUP, 
    #                         MI_CLIENT_ID, MAX_CONCURRENT, NUM_POOLS, REGION, EXECUTION_MODE)

    with open("execution_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to execution_results.json")