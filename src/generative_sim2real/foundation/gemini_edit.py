"""Closed-model editor client: the same inputs and prompt as qwen_edit, sent to the Gemini image API.

NOT executed by default. `--dry-run` is the default and makes no network call and needs no key; it
prints the plan and an estimated cost. Real calls need `--execute`, a key in the environment variable
GEMINI_API_KEY (never logged or written), and are bounded by the hard cap `--max-requests` (every
attempt, including retries, counts against it).

Request format (verified 2026-11-21 against https://aistudio.google.com/docs/image-generation):
Interactions API, client.interactions.create(model=..., input=[text, image, ...],
response_format={"type": "image", "aspect_ratio": "1:1", "image_size": "1K"}); the output image is the
base64 `interaction.output_image.data`. `--api generate_content` selects the older
client.models.generate_content path (config.image_config) as a fallback.
Prices (verified at https://aistudio.google.com/docs/pricing): see PRICING below.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

from PIL import Image

from .items import SIZE, load_items
from .prompt import PROMPT, PROMPT_DEPTH

DEFAULT_MODEL = 'gemini-nano-banana-2.1'
# USD. output per image by size; input per 1M tokens; 'batch' = batch-tier output per image.
PRICING = {
    'gemini-nano-banana-2.1': {'output': {'1K': 0.0336, '2K': 0.0504, '4K': 0.113},
                               'batch_output': {'1K': 0.0168, '2K': 0.0252, '4K': 0.0567},
                               'input_per_m': 1.50},
    'gemini-3-pro-image': {'output': {'1K': 0.134, '2K': 0.134, '4K': 0.24}, 'input_per_m': 2.00},
}
INPUT_TOKENS_PER_IMAGE = 1100  # assumption (Pro lists 560); the prompt text adds ~100 tokens
RETRIABLE = ('429', '500', '502', '503', '504', 'UNAVAILABLE', 'RESOURCE_EXHAUSTED', 'DEADLINE', 'timed out', 'APIConnectionError', 'APITimeoutError')


def estimate_cost(model, image_size, n_requests, n_input_images=3):
    if model not in PRICING:
        raise ValueError(f'no price table for {model!r}; add it to PRICING from the official page')
    table = PRICING[model]
    per_input = (n_input_images * INPUT_TOKENS_PER_IMAGE + 100) * table['input_per_m'] / 1e6
    per_request = table['output'][image_size] + per_input
    return {'per_request_usd': per_request, 'total_usd': per_request * n_requests,
            'assumed_input_tokens_per_image': INPUT_TOKENS_PER_IMAGE}


def check_output(image, size=SIZE):
    """(image resized to size x size, record). Raises on non-square (aspect drift): never crops."""
    record = {'model_output_size': list(image.size), 'aspect_ok': image.size[0] == image.size[1],
              'resized_back': False}
    if not record['aspect_ok']:
        raise ValueError(f'aspect drift: model returned {image.size}')
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
        record['resized_back'] = True
    return image, record


def _png_bytes(path):
    return Path(path).read_bytes()


def send(client, model, api, image_paths, image_size, prompt=PROMPT):
    """One request -> (PIL image, usage dict). `client` is a google.genai.Client (or a test double)."""
    if api == 'interactions':
        parts = [{'type': 'text', 'text': prompt}] + [
            {'type': 'image', 'data': base64.b64encode(_png_bytes(p)).decode(), 'mime_type': 'image/png'}
            for p in image_paths]
        interaction = client.interactions.create(
            model=model, input=parts,
            response_format={'type': 'image', 'aspect_ratio': '1:1', 'image_size': image_size})
        data = base64.b64decode(interaction.output_image.data)
        usage = getattr(interaction, 'usage', None)
        return Image.open(io.BytesIO(data)).convert('RGB'), _usage_dict(usage)
    from google.genai import types
    response = client.models.generate_content(
        model=model, contents=[prompt] + [Image.open(p).convert('RGB') for p in image_paths],
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE'],
            image_config=types.ImageConfig(aspect_ratio='1:1', image_size=image_size)))
    for part in response.parts or []:
        if getattr(part, 'inline_data', None) is not None:
            return Image.open(io.BytesIO(part.inline_data.data)).convert('RGB'), _usage_dict(
                getattr(response, 'usage_metadata', None))
    raise RuntimeError('response contained no image part')


def _usage_dict(usage):
    if usage is None:
        return None
    for attr in ('model_dump', 'to_dict'):
        if hasattr(usage, attr):
            return json.loads(json.dumps(getattr(usage, attr)(), default=str))
    return {'repr': str(usage)}


class Budget:
    """Hard cap on attempted requests (retries included)."""

    def __init__(self, cap):
        self.cap, self.used = cap, 0

    def take(self):
        if self.used >= self.cap:
            raise RuntimeError(f'request cap {self.cap} reached; stopping before any further call')
        self.used += 1


def run_one(client, budget, model, api, image_paths, image_size, retries, sleep=time.sleep, prompt=PROMPT):
    for attempt in range(retries + 1):
        budget.take()
        try:
            return send(client, model, api, image_paths, image_size, prompt), attempt
        except Exception as exc:  # noqa: BLE001  (SDK raises several types; classify by message)
            text = f'{type(exc).__name__}: {exc}'
            if attempt == retries or not any(token in text for token in RETRIABLE):
                raise
            sleep(min(60, 2 ** attempt * 4))
    raise AssertionError('unreachable')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', required=True, choices=['fixed8', 'subset300'])
    parser.add_argument('--inputs', default='outputs/foundation/inputs/inputs.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--api', default='interactions', choices=['interactions', 'generate_content'])
    parser.add_argument('--image-size', default='1K', choices=['1K', '2K', '4K'])
    parser.add_argument('--max-requests', type=int, default=10, help='hard cap on attempted requests')
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--execute', action='store_true', help='make real, billed API calls (default: dry run)')
    parser.add_argument('--condition', default='frame', choices=['frame', 'depth'],
                        help='frame: edit the SimCol frame; depth: generate from its depth control image')
    parser.add_argument('--depth-dir', default='outputs/foundation/inputs/depth300',
                        help='depth control images NNN.png (subset300 only; see foundation.inputs --depth300)')
    parser.add_argument('--every', type=int, default=1, help='use every k-th item of the set (evenly spread subset)')
    args = parser.parse_args(argv)
    items, refs = load_items(args.inputs, args.set)
    items = items[::args.every]
    prompt = PROMPT
    if args.condition == 'depth':
        if args.set != 'subset300':
            raise SystemExit('--condition depth is defined for subset300 only')
        prompt = PROMPT_DEPTH
        items = [{**i, 'real': str(Path(args.depth_dir) / f'{i["name"]}.png')} for i in items]
    out = Path(args.out)
    todo = [i for i in items if not (out / f'{i["name"]}.png').exists()]
    planned = min(len(todo), args.max_requests)
    cost = estimate_cost(args.model, args.image_size, planned)
    print(f'model={args.model} api={args.api} size={args.image_size} set={args.set} condition={args.condition}: '
          f'{len(todo)} images pending, cap {args.max_requests}, would send {planned} request(s)')
    print(f'estimated cost: ${cost["total_usd"]:.3f} (${cost["per_request_usd"]:.4f}/request, '
          f'{cost["assumed_input_tokens_per_image"]} assumed input tokens/image; retries not included)')
    if not args.execute:
        print('DRY RUN: no network call made. Re-run with --execute (and GEMINI_API_KEY set) to send.')
        for item in todo[:planned]:
            print('  would send', item['name'], '<-', [item['real'], *refs])
        return 0
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise SystemExit('GEMINI_API_KEY is not set')
    from google import genai
    client = genai.Client(api_key=key)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_requests)
    failures = 0
    for item in todo:
        if budget.used >= budget.cap:
            print('request cap reached; stopping')
            break
        paths = [item['real'], *refs]
        log = {'key': item['key'], 'sample_id': item['sample_id'], 'model': args.model, 'api': args.api,
               'image_size_requested': args.image_size, 'inputs': paths,
               'input_sha256': [hashlib.sha256(_png_bytes(p)).hexdigest() for p in paths],
               'condition': args.condition, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        start = time.time()
        try:
            (image, usage), attempt = run_one(client, budget, args.model, args.api, paths,
                                              args.image_size, args.retries, prompt=prompt)
            image, record = check_output(image)
            image.save(out / f'{item["name"]}.png')
            log.update(status='ok', attempts=attempt + 1, usage=usage, **record)
        except Exception as exc:  # noqa: BLE001  (logged without any request headers or key)
            failures += 1
            message = f'{type(exc).__name__}: {exc}'.replace(key, '***')
            log.update(status='error', error=message[:500])
        log['seconds'] = round(time.time() - start, 2)
        (out / f'{item["name"]}.json').write_text(json.dumps(log, indent=1) + '\n')
        print(item['name'], log['status'], f'{log["seconds"]}s', flush=True)
    print(f'requests attempted: {budget.used}/{budget.cap}; failures: {failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
