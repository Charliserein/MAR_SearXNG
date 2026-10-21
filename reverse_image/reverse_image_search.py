"""
Script to add captions to DGM4 test set using reverse image search.
Reads samples from test_set.json, performs reverse image search for each image,
and adds the extracted captions to the original file.

Features:
- Multi-threaded processing for parallel image search
- Batch saving every N samples to reduce I/O operations
"""
import asyncio
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

from serper_lens import SerperLensClient
from download_inverse_annotations_dirs import get_captions_and_process
from utils import get_img_content_req, compare_images


# =============================================================================
# Configuration
# =============================================================================
JSON_FILE_PATH = "../data_processing/DGM4/test_final.json"
IMAGE_BASE_PATH = "../dataset/DGM4"

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SERPER_API_KEYS = [key.strip() for key in os.getenv("MAR_SERPER_API_KEYS", "").split(",") if key.strip()]
IMGBB_API_KEYS = [key.strip() for key in os.getenv("MAR_IMGBB_API_KEYS", "").split(",") if key.strip()]

MAX_RESULTS = 20
HASHING_CUTOFF = 5
SIMILARITY_CUTOFF = 10
LOG_FILE = "reverse_dgm4.log"

# Multi-threading and batch saving configuration
MAX_WORKERS = 6  # Maximum number of parallel threads
BATCH_SAVE_INTERVAL = 100  # Save results every N samples processed

# Configure logger
logger.remove()
logger.add(
    LOG_FILE,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    encoding="utf-8"
)


# =============================================================================
# Helper functions
# =============================================================================
def load_local_image(image_path: str):
    """Load a local image file and return numpy array and PIL object."""
    try:
        from PIL import Image
        import numpy as np
        im_pil = Image.open(image_path)
        im = np.asarray(im_pil)
        return im, im_pil
    except Exception as e:
        logger.error(f"Failed to load local image {image_path}: {e}")
        return None, None


class PlaceholderPage:
    """Placeholder page object for caption extraction."""
    def __init__(self, url: str):
        self.url = url
        self.page_title = ''


def get_caption_text(caption_dict: Dict[str, str]) -> str:
    """
    Extract caption text from caption dictionary with priority:
    1. caption_node
    2. alt_node
    3. First value in caption dict
    """
    if not caption_dict:
        return ""

    if "caption_node" in caption_dict and caption_dict["caption_node"]:
        return str(caption_dict["caption_node"])

    if "alt_node" in caption_dict and caption_dict["alt_node"]:
        return str(caption_dict["alt_node"])

    if caption_dict:
        first_value = next(iter(caption_dict.values()))
        if first_value:
            return str(first_value)

    return ""


def get_concatenated_captions(results: List[Dict[str, Any]]) -> str:
    """Generate concatenated captions from search results with deduplication."""
    seen = set()
    unique_results = []

    for result in results:
        caption_dict = result.get('caption', {})
        caption_text = get_caption_text(caption_dict)
        if caption_text:
            source = result.get('source', '')
            title = result.get('title', '')
            unique_key = (source, caption_text)
            if unique_key not in seen:
                seen.add(unique_key)
                unique_results.append((source, title, caption_text))

    all_captions = []
    for idx, (source, title, caption_text) in enumerate(unique_results, start=1):
        result_lines = [f"[Re{idx}] Source: {source}"]
        if title:
            result_lines.append(f"Title: {title}")
        result_lines.append(f"Content: {caption_text}")
        all_captions.append("\n".join(result_lines))

    return "\n\n".join(all_captions)


async def process_single_image(
    image_full_path: str,
    serper_api_key: str,
    imgbb_api_key: str,
    max_results: int,
    hashing_cutoff: int,
    similarity_cutoff: int,
) -> str:
    """
    Process a single image to extract captions using reverse image search.

    Args:
        image_full_path: Full path to the image
        serper_api_key: Serper API key
        imgbb_api_key: ImgBB API key
        max_results: Maximum number of results
        hashing_cutoff: Hashing cutoff for caption extraction
        similarity_cutoff: Image similarity cutoff

    Returns:
        Concatenated caption string or empty string if failed
    """
    try:
        # Load the original image
        original_img, original_img_pil = load_local_image(image_full_path)
        if original_img is None:
            logger.warning(f"Failed to load image: {image_full_path}")
            return ""

        # Initialize client and perform search
        client = SerperLensClient(api_key=serper_api_key, max_results=max_results)
        search_results = await client.search(
            image_path=Path(image_full_path),
            imgbb_api_key=imgbb_api_key,
        )

        if not search_results:
            logger.warning(f"No search results for: {image_full_path}")
            return ""

        # Log search results
        logger.info(f"Found {len(search_results)} search results for: {image_full_path}")
        for idx, match_item in enumerate(search_results, start=1):
            logger.info(f"  Result [{idx}]:")
            logger.info(f"    Source: {match_item.source}")
            logger.info(f"    Page URL: {match_item.link}")
            logger.info(f"    Image URL: {match_item.image_url}")

        # Process search results
        captions = []
        seen = set()

        for match_item in search_results:
            img_url = match_item.image_url
            page_url = match_item.link
            source = match_item.source

            # Check image similarity
            result_img, result_img_pil = get_img_content_req(img_url)
            if isinstance(result_img, int) and result_img == 0:
                continue

            is_similar, actual_diff = compare_images(
                original_img, original_img_pil, result_img, result_img_pil, similarity_cutoff
            )
            if not is_similar:
                continue

            # Extract caption
            page = PlaceholderPage(url=page_url)
            entry = get_captions_and_process(
                img_url=img_url,
                page_url=page_url,
                page=page,
                save_folder_path="/tmp/captions",  # Temporary folder
                file_save_counter=0,
                hashing_cutoff=hashing_cutoff,
            )

            if entry:
                caption_dict = entry.get('caption', {})
                caption_text = get_caption_text(caption_dict)
                if caption_text:
                    title = entry.get('title', '')
                    unique_key = (source, caption_text)
                    if unique_key not in seen:
                        seen.add(unique_key)
                        result_lines = [f"[Re{len(captions) + 1}] Source: {source}"]
                        if title:
                            result_lines.append(f"Title: {title}")
                        result_lines.append(f"Content: {caption_text}")
                        captions.append("\n".join(result_lines))
                        logger.info(f"  Extracted caption from {page_url}")
                        logger.info(f"    Caption text: {caption_text[:100]}{'...' if len(caption_text) > 100 else ''}")

        final_captions = "\n\n".join(captions) if captions else ""
        if final_captions:
            logger.info(f"Final concatenated captions ({len(final_captions)} chars):")
            logger.info(f"{final_captions}")

        return final_captions

    except Exception as e:
        logger.error(f"Error processing image {image_full_path}: {e}")
        return ""


def save_json_thread_safe(data: List[Dict], json_file_path: str, lock: threading.Lock) -> None:
    """Thread-safe JSON saving function."""
    with lock:
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")


async def process_single_sample(
    idx: int,
    sample: Dict[str, Any],
    image_base_path: str,
    serper_api_keys: List[str],
    imgbb_api_keys: List[str],
    max_results: int,
    hashing_cutoff: int,
    similarity_cutoff: int,
) -> Tuple[int, str]:
    """
    Process a single sample and return its index and extracted captions.

    Args:
        idx: Index of the sample in the data list
        sample: Sample dictionary
        image_base_path: Base path for images
        serper_api_keys: List of Serper API keys (one will be randomly selected)
        imgbb_api_keys: List of ImgBB API keys (one will be randomly selected)
        max_results: Maximum number of search results
        hashing_cutoff: Hashing cutoff for caption extraction
        similarity_cutoff: Image similarity cutoff

    Returns:
        Tuple of (index, captions)
    """
    image_path = sample.get('image_path', '')
    if not image_path:
        logger.warning(f"Sample {idx} has no image_path, skipping")
        return idx, ""

    # Construct full image path
    image_full_path = os.path.join(image_base_path, image_path)

    if not os.path.exists(image_full_path):
        logger.warning(f"Image not found: {image_full_path}, skipping")
        return idx, ""

    logger.info(f"Processing [{idx + 1}]: {image_path}")

    # Randomly select one API key for this sample
    serper_api_key = random.choice(serper_api_keys)
    imgbb_api_key = random.choice(imgbb_api_keys)

    # Extract captions
    captions = await process_single_image(
        image_full_path,
        serper_api_key,
        imgbb_api_key,
        max_results,
        hashing_cutoff,
        similarity_cutoff,
    )

    if captions:
        logger.info(f"  [{idx + 1}] Successfully extracted captions ({len(captions)} chars)")
        logger.info(f"  [{idx + 1}] Captions to save in JSON:")
        logger.info(f"{captions}")
    else:
        logger.warning(f"  [{idx + 1}] No captions extracted")

    return idx, captions


async def add_captions_to_json(
    json_file_path: str,
    image_base_path: str,
    serper_api_keys: List[str],
    imgbb_api_keys: List[str],
    max_results: int = 10,
    hashing_cutoff: int = 5,
    similarity_cutoff: int = 10,
    max_workers: int = MAX_WORKERS,
    batch_save_interval: int = BATCH_SAVE_INTERVAL,
) -> None:
    """
    Add captions to all samples in the JSON file using multi-threading.

    Args:
        json_file_path: Path to the JSON file
        image_base_path: Base path for images
        serper_api_keys: List of Serper API keys (one randomly selected per sample)
        imgbb_api_keys: List of ImgBB API keys (one randomly selected per sample)
        max_results: Maximum number of search results
        hashing_cutoff: Hashing cutoff for caption extraction
        similarity_cutoff: Image similarity cutoff
        max_workers: Maximum number of parallel threads
        batch_save_interval: Save results every N samples processed
    """
    # Load JSON data
    logger.info(f"Loading JSON file: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    logger.info(f"Total samples to process: {total_samples}")
    logger.info(f"Using {max_workers} worker threads")
    logger.info(f"Batch save interval: {batch_save_interval} samples")

    # Thread lock for safe JSON writing
    save_lock = threading.Lock()
    processed_count = 0

    # Create samples to process (filter out samples without image_path or invalid images)
    tasks_to_process = []
    for idx, sample in enumerate(data):
        image_path = sample.get('image_path', '')
        if not image_path:
            continue

        image_full_path = os.path.join(image_base_path, image_path)
        if not os.path.exists(image_full_path):
            continue

        tasks_to_process.append((idx, sample))

    logger.info(f"Valid samples to process: {len(tasks_to_process)}")

    # Process samples in batches using ThreadPoolExecutor
    loop = asyncio.get_event_loop()

    with tqdm(total=len(tasks_to_process), desc="Processing images", unit="img") as pbar:
        for batch_start in range(0, len(tasks_to_process), max_workers):

            batch_end = min(batch_start + max_workers, len(tasks_to_process))
            batch = tasks_to_process[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start // max_workers + 1}: samples {batch_start}-{batch_end - 1}")

            async_tasks = []
            for idx, sample in batch:
                task = process_single_sample(
                    idx, sample, image_base_path, serper_api_keys, imgbb_api_keys,
                    max_results, hashing_cutoff, similarity_cutoff
                )
                async_tasks.append(task)

            batch_results = await asyncio.gather(*async_tasks)

            for idx, captions in batch_results:
                data[idx]['captions'] = captions
                processed_count += 1

                # Update progress bar
                pbar.set_postfix({"saved": f"{processed_count}"})
                pbar.update(1)

            # Batch save: save every batch_save_interval samples
            if processed_count % batch_save_interval == 0:
                pbar.set_description(f"Processing images (saving...)")
                await loop.run_in_executor(
                    None,
                    save_json_thread_safe,
                    data, json_file_path, save_lock
                )
                pbar.set_description("Processing images")

    # Final save
    logger.info(f"Saving final results to: {json_file_path}")
    save_json_thread_safe(data, json_file_path, save_lock)
    logger.info("Done!")


@logger.catch()
async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Adding Captions to AMG_MEDIA Test Set")
    logger.info("Multi-threaded processing enabled")
    logger.info("=" * 60)

    await add_captions_to_json(
        json_file_path=JSON_FILE_PATH,
        image_base_path=IMAGE_BASE_PATH,
        serper_api_keys=SERPER_API_KEYS,
        imgbb_api_keys=IMGBB_API_KEYS,
        max_results=MAX_RESULTS,
        hashing_cutoff=HASHING_CUTOFF,
        similarity_cutoff=SIMILARITY_CUTOFF,
        max_workers=MAX_WORKERS,
        batch_save_interval=BATCH_SAVE_INTERVAL,
    )


if __name__ == "__main__":
    asyncio.run(main())
