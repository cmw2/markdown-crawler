from bs4 import BeautifulSoup
import urllib.parse
import threading
from markdownify import markdownify as md
import requests
import logging
import queue
import time
import os
import re
from typing import (
    List,
    Optional,
    Union
)
__version__ = '0.1'
__author__ = 'Paul Pierre (github.com/paulpierre)'
__copyright__ = "(C) 2023 Paul Pierre. MIT License."
__contributors__ = ['Paul Pierre']

BANNER = r"""
                |                                     |             
 __ `__ \    _` |        __|   __|   _` | \ \  \   /  |   _ \   __| 
 |   |   |  (   |       (     |     (   |  \ \  \ /   |   __/  |    
_|  _|  _| \__._|      \___| _|    \__._|   \_/\_/   _| \___| _|    

-------------------------------------------------------------------------
A multithreaded 🕸️ web crawler that recursively crawls a website and
creates a 🔽 markdown file for each page by https://github.com/paulpierre
-------------------------------------------------------------------------
"""

logger = logging.getLogger(__name__)
DEFAULT_BASE_DIR = 'markdown'
DEFAULT_MAX_DEPTH = 3
DEFAULT_NUM_THREADS = 5
DEFAULT_TARGET_CONTENT = ['article', 'div', 'main', 'p']
DEFAULT_TARGET_LINKS = ['body']
DEFAULT_DOMAIN_MATCH = True
DEFAULT_BASE_PATH_MATCH = True


# --------------
# URL validation
# --------------
def is_valid_url(url: str) -> bool:
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        logger.debug(f'❌ Invalid URL {url}')
        return False


# ----------------
# Clean up the URL
# ----------------
def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), None, None, None))


# ------------------
# HTML parsing logic
# ------------------
def crawl(
    url: str,
    base_url: str,
    file_path: str,
    worker_index: int = -1,
    files_created: list = None,
    file_counter_lock: threading.Lock = None,
    target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
    target_content: Union[str, List[str]] = None,
    valid_paths: Union[str, List[str]] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH,
    is_links: Optional[bool] = False
) -> List[str]:
        
    try:
        logger.info(f'[T{worker_index}] Crawling: {url}')
        
        # Headers to mimic a real browser and avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Check domain before making request if domain matching is required
        if is_domain_match:
            url_domain = urllib.parse.urlparse(url).netloc
            base_domain = urllib.parse.urlparse(base_url).netloc
            if url_domain != base_domain:
                logger.debug(f'[T{worker_index}] Skipping {url} - domain {url_domain} does not match base domain {base_domain}')
                return []
        
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=False)
        
        # Handle redirects manually to check domain on final URL
        if response.status_code in [301, 302, 303, 307, 308]:
            redirect_url = response.headers.get('Location')
            if redirect_url:
                # Make redirect URL absolute
                redirect_url = urllib.parse.urljoin(url, redirect_url)
                logger.debug(f'[T{worker_index}] Redirect from {url} to {redirect_url}')

                # Check if redirect stays within allowed domain
                if is_domain_match:
                    redirect_domain = urllib.parse.urlparse(redirect_url).netloc
                    base_domain = urllib.parse.urlparse(base_url).netloc
                    if redirect_domain != base_domain:
                        logger.debug(f'[T{worker_index}] Skipping redirect to {redirect_url} - domain {redirect_domain} does not match base domain {base_domain}')
                        return []
                
                # Follow the redirect
                response = requests.get(redirect_url, headers=headers, timeout=10)
                # Update url to the final redirected URL for further processing
                url = redirect_url
                
    except requests.exceptions.RequestException as e:
        logger.error(f'[T{worker_index}]  ❌ Request error for {url}: {e}')
        return []
    
    # Check for non-successful status codes
    if response.status_code != 200:
        if response.status_code == 403:
            logger.error(f'[T{worker_index}]  ❌ Access forbidden (403) for {url}')
            logger.debug(f'[T{worker_index}] Response headers: {dict(response.headers)}')
            if response.text:
                logger.debug(f'[T{worker_index}] Response body: {response.text[:500]}...' if len(response.text) > 500 else f'Response body: {response.text}')
        elif response.status_code == 404:
            logger.error(f'[T{worker_index}]  ❌ Page not found (404) for {url}')
        elif response.status_code == 429:
            logger.error(f'[T{worker_index}]  ❌ Rate limited (429) for {url}')
            logger.debug(f'[T{worker_index}] Response headers: {dict(response.headers)}')
        else:
            logger.error(f'[T{worker_index}] ❌ HTTP {response.status_code} error for {url}')
            logger.debug(f'[T{worker_index}] Response headers: {dict(response.headers)}')
            if response.text:
                logger.debug(f'[T{worker_index}] Response body: {response.text[:500]}...' if len(response.text) > 500 else f'Response body: {response.text}')
        return []
    
    if 'text/html' not in response.headers.get('Content-Type', ''):
        logger.error(f'[T{worker_index}] ❌ Content not text/html for {url}')
        return []
    
    # List of elements we want to strip
    # ---------------------------------
    strip_elements = []

    if is_links:
        strip_elements = ['a']

    # -------------------------------
    # Create BS4 instance for parsing
    # -------------------------------
    soup = BeautifulSoup(response.text, 'html.parser')

    # Strip unwanted tags
    for script in soup(['script', 'style']):
        script.decompose()

    # --------------------------------------------
    # Write the markdown file if it does not exist
    # --------------------------------------------
    if not os.path.exists(file_path):

        file_name = file_path.split("/")[-1]

        # ------------------
        # Get target content
        # ------------------

        content = get_target_content(soup, target_content=target_content)

        if content:
            # --------------
            # Parse markdown
            # --------------
            output = md(
                content,
                keep_inline_images_in=['td', 'th', 'a', 'figure'],
                strip=strip_elements
            )

            # ------------------------------
            # Write markdown content to file
            # ------------------------------
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(output)

            # Increment file counter thread-safely
            if files_created is not None and file_counter_lock is not None:
                with file_counter_lock:
                    files_created[0] += 1
                    
            logger.info(f'[T{worker_index}] Created 📝 {file_name}')
        else:
            logger.error(f'[T{worker_index}] ❌ Empty content for {file_path}. Target selectors: {target_content}')
            #logger.debug(f'Available elements on page: {[tag.name for tag in soup.find_all()][:20]}')  # Show first 20 element types

    child_urls = get_target_links(
        soup,
        base_url,
        target_links,
        valid_paths=valid_paths,
        is_domain_match=is_domain_match,
        is_base_path_match=is_base_path_match    
    )

    logger.debug(f'[T{worker_index}] Found {len(child_urls) if child_urls else 0} child URLs')
    return child_urls


def get_target_content(
    soup: BeautifulSoup,
    target_content: Union[List[str], None] = None
) -> str:

    content = ''

    # -------------------------------------
    # Get target content by target selector
    # -------------------------------------
    if target_content:
        for target in target_content:
            # Use CSS selector instead of find_all for more flexibility
            elements = soup.select(target)
            if elements:
                for element in elements:
                    content += f'{str(element)}'.replace('\n', '')
            # else:
            #     logger.debug(f'No elements found for selector: {target}')

    # ---------------------------
    # Naive estimation of content
    # ---------------------------
    else:
        max_text_length = 0
        main_content = None
        for tag in soup.find_all(DEFAULT_TARGET_CONTENT):
            text_length = len(tag.get_text())
            if text_length > max_text_length:
                max_text_length = text_length
                main_content = tag

        if main_content:
            content = str(main_content)

    return content if len(content) > 0 else False


def get_target_links(
    soup: BeautifulSoup,
    base_url: str,
    target_links: List[str] = DEFAULT_TARGET_LINKS,
    valid_paths: Union[List[str], None] = None,
    is_domain_match: Optional[bool] = DEFAULT_DOMAIN_MATCH,
    is_base_path_match: Optional[bool] = DEFAULT_BASE_PATH_MATCH
) -> List[str]:

    child_urls = []

    # Get all urls from target_links
    for target in soup.find_all(target_links):
        # Get all the links in target
        for link in target.find_all('a'):
            child_urls.append(urllib.parse.urljoin(base_url, link.get('href')))

    result = []
    for u in child_urls:

        child_url = urllib.parse.urlparse(u)

        # ---------------------------------
        # Check if domain match is required
        # ---------------------------------
        if is_domain_match and child_url.netloc != urllib.parse.urlparse(base_url).netloc:
            continue

        if is_base_path_match and child_url.path.startswith(urllib.parse.urlparse(base_url).path):
            result.append(u)
            continue

        if valid_paths:
            for valid_path in valid_paths:
                if child_url.path.startswith(urllib.parse.urlparse(valid_path).path):
                    result.append(u)
                    break

    return result


# ------------------
# Worker thread logic
# ------------------
def worker(
    q: object,
    base_url: str,
    max_depth: int,
    queued_urls: set,
    url_lock: threading.Lock,
    base_dir: str,
    stop_flag: threading.Event,
    active_workers: list,
    worker_index: int,
    files_created: list,
    file_counter_lock: threading.Lock,
    target_links: Union[List[str], None] = DEFAULT_TARGET_LINKS,
    target_content: Union[List[str], None] = None,
    valid_paths: Union[List[str], None] = None,
    is_domain_match: bool = None,
    is_base_path_match: bool = None,
    is_links: Optional[bool] = False
) -> None:

    while not stop_flag.is_set():
        try:
            depth, url = q.get(timeout=2)  # Add timeout to allow checking stop_flag
            active_workers[worker_index] = True  # Mark this worker as active
        except queue.Empty:
            # Queue is empty, mark this worker as inactive
            active_workers[worker_index] = False
            
            # Check if all workers are inactive (queue empty and no one working)
            if not any(active_workers):
                logger.debug(f'[T{worker_index}] Worker {worker_index} detected all workers idle, stopping')
                stop_flag.set()  # Signal all threads to stop
                break
            
            # Other workers might still be processing, so continue waiting
            continue
            
        if depth > max_depth or stop_flag.is_set():
            logger.debug(f'[T{worker_index}] Skipping {url} at depth {depth} (max depth {max_depth})')
            continue
            
        # Create a more unique filename using the full URL including domain
        parsed_url = urllib.parse.urlparse(url)
        
        # Start with the domain (netloc)
        domain_parts = parsed_url.netloc.split('.')
        path_parts = [part for part in parsed_url.path.split('/') if part]  # Remove empty parts
        
        # Combine domain and path parts
        all_parts = domain_parts + path_parts
        
        if all_parts:
            # Join all parts with underscores and clean for filesystem
            file_name = '_'.join(all_parts)
            # Remove or replace invalid filename characters
            file_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
            # Replace multiple underscores with single ones
            file_name = re.sub(r'_+', '_', file_name)
        else:
            file_name = 'index'
            
        file_path = f'{base_dir.rstrip("/") + "/"}{file_name}.md'

        # URL is guaranteed to be unique from queue.Queue() thread safety
        # No need for additional duplicate checking here
        
        child_urls = crawl(
            url,
            base_url,
            file_path,
            worker_index,
            files_created,
            file_counter_lock,
            target_links,
            target_content,
            valid_paths,
            is_domain_match,
            is_base_path_match,
            is_links
        )
        child_urls = [normalize_url(u) for u in child_urls]
        
        # Check if we can add child URLs (not at max depth yet)
        if depth < max_depth:
            # Only add URLs to queue that haven't been seen before
            with url_lock:
                for child_url in child_urls:
                    if not stop_flag.is_set() and child_url not in queued_urls:
                        q.put((depth + 1, child_url))
                        queued_urls.add(child_url)  # Mark as seen to prevent duplicates
                        logger.debug(f'[T{worker_index}] Added to queue: {child_url} at depth {depth + 1}')
                    else:
                        logger.debug(f'[T{worker_index}] Skipping already seen URL: {child_url}')
        else:
            logger.debug(f'[T{worker_index}] Not adding any URLs - at max depth {max_depth}')
        # Mark worker as inactive after processing
        active_workers[worker_index] = False
        
        # Small delay to prevent overwhelming the server, but much shorter
        time.sleep(0.5)


# -----------------
# Thread management
# -----------------
def md_crawl(
        base_url: str,
        max_depth: Optional[int] = DEFAULT_MAX_DEPTH,
        num_threads: Optional[int] = DEFAULT_NUM_THREADS,
        base_dir: Optional[str] = DEFAULT_BASE_DIR,
        target_links: Union[str, List[str]] = DEFAULT_TARGET_LINKS,
        target_content: Union[str, List[str]] = None,
        valid_paths: Union[str, List[str]] = None,
        is_domain_match: Optional[bool] = None,
        is_base_path_match: Optional[bool] = None,
        is_debug: Optional[bool] = False,
        is_links: Optional[bool] = False
) -> None:
    if is_domain_match is False and is_base_path_match is True:
        raise ValueError('❌ Domain match must be True if base match is set to True')

    is_domain_match = DEFAULT_DOMAIN_MATCH if is_domain_match is None else is_domain_match
    is_base_path_match = DEFAULT_BASE_PATH_MATCH if is_base_path_match is None else is_base_path_match

    if not base_url:
        raise ValueError('❌ Base URL is required')

    if isinstance(target_links, str):
        target_links = target_links.split(',') if ',' in target_links else [target_links]

    if isinstance(target_content, str):
        target_content = target_content.split(',') if ',' in target_content else [target_content]

    if isinstance(valid_paths, str):
        valid_paths = valid_paths.split(',') if ',' in valid_paths else [valid_paths]

    if is_debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug('🐞 Debugging enabled')
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f'🕸️ Crawling {base_url} at ⏬ depth {max_depth} with 🧵 {num_threads} threads')
    
    # Record start time for elapsed time calculation
    start_time = time.time()

    # Validate the base URL
    if not is_valid_url(base_url):
        raise ValueError('❌ Invalid base URL')

    # Create base_dir if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    queued_urls = set()  # Track URLs that have been seen/queued
    url_lock = threading.Lock()  # Lock for thread-safe access to queued_urls
    stop_flag = threading.Event()
    active_workers = [False] * num_threads  # Track which workers are currently processing
    
    # File creation counter and lock for thread-safe access
    files_created = [0]  # Use list to make it mutable for threads
    file_counter_lock = threading.Lock()

    # Create a queue of URLs to crawl
    q = queue.Queue()

    # Add the base URL to the queue
    q.put((0, base_url))
    queued_urls.add(base_url)  # Track that base URL is queued

    threads = []

    # Create a thread for each URL in the queue
    for i in range(num_threads):
        t = threading.Thread(
            target=worker,
            args=(
                q,
                base_url,
                max_depth,
                queued_urls,
                url_lock,
                base_dir,
                stop_flag,
                active_workers,
                i,  # worker index
                files_created,
                file_counter_lock,
                target_links,
                target_content,
                valid_paths,
                is_domain_match,
                is_base_path_match,
                is_links
            )
        )
        t.daemon = True  # Make thread daemon so it dies when main thread exits
        threads.append(t)
        t.start()
        logger.debug(f'Started thread {i+1} of {num_threads}')

    try:
        # Wait for all threads to finish with timeout to allow interrupts
        for t in threads:
            while t.is_alive():
                t.join(timeout=5)  # Check every five seconds to allow KeyboardInterrupt
    except KeyboardInterrupt:
        logger.info('🛑 Interrupted by user, stopping crawl...')
        stop_flag.set()
        
        # Give threads a moment to stop gracefully
        for t in threads:
            if t.is_alive():
                t.join(timeout=2)
                if t.is_alive():
                    logger.warning(f'Thread {t.name} did not exit gracefully')

    # Calculate and log elapsed time and file count
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    total_files = files_created[0]
    
    if minutes > 0:
        logger.info(f'🏁 All threads have finished - Total time: {int(minutes)}m {seconds:.1f}s - Files created: {total_files}')
    else:
        logger.info(f'🏁 All threads have finished - Total time: {elapsed_time:.1f}s - Files created: {total_files}')