import json
import os
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import result

import torch
from NewsAnalyzer import NewsAnalyzer
from retrieval.provenance import evidence_kind
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def process_sample(sample, dataset_type, analyzer, image_base_path, logger):
    try:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text = sample.get('text', '')
        image_path = sample.get('image_path', '')
        captions = sample.get('captions', '')

        if image_path:
            image_path = os.path.join(image_base_path, image_path)

        result = analyzer.analyze(text, image_path, captions, evidence_kind=evidence_kind(sample))

        # Record the analysis results in the log
        logger.info("=== Sample Text ===")
        logger.info(text)
        logger.info("\n=== Domain ===")
        logger.info(result["domain_tags"])
        logger.info("\n=== Domain Profile ===")
        logger.info(result["domain_profile"])
        logger.info("\n=== Query List ===")
        logger.info(result["query_list"])
        logger.info("\n=== External Knowledge ===")
        logger.info(result["external_knowledge"])
        logger.info("=== Debate Log ===")
        logger.info(result["debate_log"])
        logger.info("\n=== Short Term Memories ===")
        logger.info(result["short_term_memories"])
        logger.info("\n=== Long Term Memory ===")
        logger.info(result["long_term_memory"])
        logger.info("\n=== Debate Summary ===")
        logger.info(result["debate_summary"])
        logger.info("\n=== Final Label ===")
        logger.info(result["label"])
        logger.info(result["text_reason"])
        logger.info(result["image_reason"])
        logger.info(result["consistency_reason"])
        logger.info("=" * 80)


        sample['label'] = result['label']
        sample['text_reason'] = result['text_reason']
        sample['image_reason'] = result['image_reason']
        sample['consistency_reason'] = result['consistency_reason']

        return sample
    except Exception as e:
        logger.error(f"Error ({dataset_type} dataset):{e}")
        logger.error(f"Image path: {image_path}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def getFileLogger(log_file):

    logger = logging.getLogger()
    if not logger.handlers:
        logger.setLevel(level = logging.INFO)

        class NameFilter(logging.Filter):
            def filter(self, record):
                return not record.name.startswith('httpx')

        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        handler.addFilter(NameFilter())

        logger.addHandler(handler)

        httpx_logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.WARNING)

    return logger

def main():

    parser = argparse.ArgumentParser(description='News_Analyzer')
    parser.add_argument('--base_url', type=str, default='http://localhost:19887/v1',
                       help='base url')
    parser.add_argument('--api_key', type=str, default='sk-123456',
                       help='api key')
    parser.add_argument('--model', type=str, default='Qwen3-VL-32B-Instruct-FP8',
                       help='model name')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='temperature')
    parser.add_argument('--log_filename', type=str, default='mmfakebench_32b.log',
                       help='log filename')
    parser.add_argument('--max_workers', type=int, default=25,
                       help='max_workers')
    parser.add_argument('--dataset_type', type=str, choices=['train', 'test'], default='test',
                       help='dataset_type: train or test')
    parser.add_argument('--output_dir', type=str,
                       default='../dataset_final/MMFakeBench',
                       help='output dir')
    parser.add_argument('--image_base_path', type=str,
                       default='../dataset/MMFakeBench/',
                       help='image base path')

    args = parser.parse_args()

    logger = getFileLogger(args.log_filename)

    analyzer = NewsAnalyzer(base_url=args.base_url, api_key=args.api_key,
                           model=args.model, temperature=args.temperature, n_rounds=3, logger=logger)

    dataset_type = args.dataset_type
    input_file_map = {
        "train": '../data_processing/MMFakeBench/train_set.json',
        "test": '../data_processing/MMFakeBench/test_set.json'
    }
    # input_file_map = {
    #     "train": '../data_processing/DGM4/train_final.json',
    #     "test": '../data_processing/DGM4/test_final.json'
    # }
    input_file = input_file_map[dataset_type]

    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    image_base_path = args.image_base_path

    output_file = os.path.join(output_dir, f"{dataset_type}_mmfakebench_32b.json")

    samples = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        try:
            data = json.load(f_in)
            if isinstance(data, list):
                samples = data
            else:
                logger.error(f"The expected input was a JSON array, but instead we received {type(data)}")
                return
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error {dataset_type} dataset: {e}")
            return

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:

        future_to_sample = {executor.submit(process_sample, sample, dataset_type, analyzer, image_base_path, logger): sample for sample in samples}

        # tqdm
        with tqdm(total=len(samples), desc=f"Processing {dataset_type} dataset") as pbar:

            with open(output_file, 'w', encoding='utf-8') as f_out:

                for future in as_completed(future_to_sample):
                    result = future.result()
                    if result:
                        json.dump(result, f_out, ensure_ascii=False)
                        f_out.write('\n')
                        f_out.flush()
                    pbar.update(1)

    logger.info(f"{dataset_type} The dataset processing is completed and the results have been saved to {output_file}")

    logger.info("The dataset processing is complete!")

if __name__ == "__main__":
    main()
