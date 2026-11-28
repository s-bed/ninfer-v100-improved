#!/usr/bin/env python3
"""Qualify a single ready production-configured server. Restart externally between phases.

Use distinct, private output and Qwen persistence paths for each binary. The Qwen
restart phase reads the generated IDs from the first process for exact comparison.
"""

import argparse
import base64
import http.client
import json
import math
from pathlib import Path
import struct
import time
import zlib
from urllib.parse import urlsplit


def require(test, message):
    if not test:
        raise RuntimeError(message)


def solid_image(rgb):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    rows = (b'\0' + bytes(rgb) * 64) * 64
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 64, 64, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


class Probe:
    def __init__(self, args):
        self.args = args
        self.base = urlsplit(args.url)
        self.out = args.output / args.phase
        self.out.mkdir(parents=True, exist_ok=True)

    def save(self, name, value):
        (self.out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')

    def connect(self, path, payload=None):
        conn = http.client.HTTPConnection(self.base.hostname, self.base.port, timeout=720)
        body = json.dumps(payload).encode() if payload is not None else None
        conn.request('POST' if body is not None else 'GET', self.base.path + path, body,
                     {'Content-Type': 'application/json'})
        response = conn.getresponse()
        if response.status != 200:
            error = response.read(2048)
            conn.close()
            raise RuntimeError(f'{path} returned {response.status}: {error!r}')
        return conn, response

    def request(self, path, payload=None):
        conn, response = self.connect(path, payload)
        try:
            return json.loads(response.read())
        finally:
            conn.close()

    def stats(self, name):
        stats = self.request('/stats')
        self.save(name + '-stats', stats)
        require(stats['model'] == self.args.model, 'unexpected served model')
        return stats['context_cache']['hybrid']

    def tokens(self, text):
        return self.request('/tokenize', dict(content=text, add_special=False))['tokens']

    def prompt(self, name, length, question):
        start = self.tokens(name + '\n')
        filler = self.tokens(' Mira keeps the astronomy notes in a blue notebook.')
        tail = self.tokens('\n' + question + '\nAnswer:')
        count = length - len(start) - len(tail)
        require(count > 0, 'prompt too short')
        return start + (filler * math.ceil(count / len(filler)))[:count] + tail

    def complete(self, name, prompt, cache=True, count=12):
        result = self.request('/completion', dict(model=self.args.model, prompt=prompt, temperature=0,
                              seed=1729, n_predict=count, cache_prompt=cache, return_tokens=True))
        self.save(name, {k: result[k] for k in ('content', 'tokens', 'tokens_cached', 'tokens_evaluated', 'timings')})
        require(result['tokens'] and result['tokens_evaluated'] == len(prompt), 'missing generated IDs or prompt truncated')
        return result['tokens'], result['tokens_cached']

    def chat(self, name, parts):
        body = dict(model=self.args.model, messages=[dict(role='user', content=parts)], temperature=0,
                    seed=1729, max_tokens=64, reasoning_effort='none')
        result = self.request('/v1/chat/completions', body)
        self.save(name, result)
        msg = result['choices'][0]['message']
        return msg.get('content') or '', result['usage']['prompt_tokens_details']['cached_tokens']

    def image(self, name, color, padding=''):
        rgb = (255, 0, 0) if color == 'red' else (0, 0, 255)
        parts = [dict(type='image_url', image_url=dict(url=solid_image(rgb))),
                 dict(type='text', text=padding + ' What color fills the image? Answer with the single color word.')]
        answer, reused = self.chat(name, parts)
        require(color in answer.strip().lower(), f'{name}: expected {color}, got {answer!r}')
        return answer, reused

    def cold(self):
        h = self.stats('startup')
        for key in ('tree_blocks', 'snapshots', 'host_used_bytes', 'host_block_restores', 'reused_tokens'):
            require(h[key] == 0, f'cold process has {key}={h[key]}')
        require(19 * 2**30 < h['host_capacity_bytes'] <= 20 * 2**30, 'incorrect host capacity')

    def inkling(self, short=False):
        self.cold()
        cap = self.request('/v1/load')['capacity']
        props = self.request('/props')
        self.save('properties', props)
        require(cap['max_context'] == cap['kv_capacity_tokens'] == 1048576 and props['modalities']['vision'],
                'Inkling context or vision missing')
        size, expected = (149, 64) if short else (35071, 32768)
        a = self.prompt('The alpine observatory', size, 'Name the color of the notebook.')
        b = self.prompt('The quiet harbor', 181, 'Name the largest ocean.')
        generated, hit = self.complete('cold', a)
        require(hit == 0, 'first request was warm')
        cold_state = self.stats('after-cold')
        self.complete('displace', b)
        before = self.stats('displaced')
        require(before['evicted_blocks'] - cold_state['evicted_blocks'] >= expected // 64,
                'unrelated conversation did not displace the expected device blocks')
        again, reused = self.complete('restored', a)
        after = self.stats('restored')
        require(again == generated and reused == expected, f'host replay returned {reused} reused tokens, expected {expected}')
        require(after['host_block_restores'] - before['host_block_restores'] == expected // 64 and
                after['host_image_restores'] > before['host_image_restores'] and
                after['reused_tokens'] - before['reused_tokens'] == expected, 'host restore counters disagree')
        if short:
            return
        branches = []
        for label, question in [('left', '\nCount in words: one, two, three, four,'),
                                ('right', '\nLetters in order: A, B, C, D,')]:
            branch = a[:expected] + self.tokens(question)
            warm, hit = self.complete(label, branch)
            cold, bypass = self.complete(label + '-uncached', branch, False)
            require(hit == expected and bypass == 0 and warm == cold, label + ' branch differs from cold')
            branches.append(warm)
        require(branches[0] != branches[1], 'branch controls yielded indistinguishable greedy output')
        self.image('vision-red', 'red')
        # Both images occupy the same token positions. Put the 8192 boundary after the image.
        pad = ' Note.' * 4300
        red, _ = self.image('vision-red-long', 'red', pad)
        repeat, red_hit = self.image('vision-red-repeat', 'red', pad)
        blue, blue_hit = self.image('vision-blue-changed', 'blue', pad)
        require(red == repeat and red_hit >= 8192 and blue_hit < red_hit,
                'changed image reused image-dependent state')
        # Compare against a separate, cold process in the reload phase.
        self.save('blue-reference', blue)
        self.cancel()
        text, _ = self.chat('text', 'Reply with exactly the uppercase word ORANGE.')
        require('ORANGE' in text.upper(), f'Inkling text response: {text!r}')

    def verify_drafter(self):
        records = [json.loads(line) for line in self.args.request_log.read_text().splitlines()]
        rounds = [r['speculative'] for r in records if r.get('speculative', {}).get('rounds', 0)]
        require(any(r['backend'] == 'dspark' and r['draft_window'] == 3 and r['drafted_tokens'] > 0
                    for r in rounds), 'no completed DSpark verification round with three draft tokens')
        self.save('drafter', rounds)

    def cancel(self):
        prompt = self.prompt('The cinnamon lighthouse', 126, 'Continue the story in many paragraphs.')
        before = self.stats('before-cancel')
        offset = self.args.request_log.stat().st_size
        conn, response = self.connect('/completion', dict(model=self.args.model, prompt=prompt, temperature=0,
                                      n_predict=768, ignore_eos=True, stream=True, return_tokens=True))
        try:
            for line in response:
                if line.startswith(b'data:'):
                    event = json.loads(line[5:])
                    if event.get('content') and not event.get('stop'):
                        break
            else:
                raise RuntimeError('cancel test received no generated chunk')
        finally:
            response.close()
            conn.close()
        end = time.monotonic() + 45
        while self.request('/stats')['requests']['running']:
            require(time.monotonic() < end, 'disconnected generation still running')
            time.sleep(.1)
        while True:
            with self.args.request_log.open('rb') as log:
                log.seek(offset)
                records = [json.loads(line) for line in log if line.endswith(b'\n')]
                closed = [r for r in records if r.get('event') == 'request_error' and
                          r.get('request', {}).get('stream') and r['request'].get('requested_output_tokens') == 768 and
                          r.get('error', {}).get('message') == 'client disconnected']
            if closed:
                break
            require(time.monotonic() < end, 'server did not record the generation cancellation')
            time.sleep(.1)
        self.save('cancel-event', closed)
        after = self.stats('after-cancel')
        require(after['tree_blocks'] - before['tree_blocks'] == 1 and
                after['snapshots'] - before['snapshots'] == 1, 'generated tokens entered the prompt cache')
        fork = prompt + self.tokens(' An entirely different ending.')
        out, hit = self.complete('after-cancel-branch', fork)
        ref, _ = self.complete('after-cancel-uncached', fork, False)
        require(hit == 64 and out == ref, 'cancelled generation contaminated a later continuation')

    def qwen(self, restored=False):
        original = self.prompt('The amber coast', 4219, 'Name the color of the notebook.')
        prompt = original + self.tokens(' One more question: what does Mira keep there?')
        ref_file = self.args.output / 'qwen-reference.json'
        if restored:
            before = self.stats('from-disk')
            require(before['snapshots'] and before['tree_blocks'] and before['host_used_bytes'], 'empty restored cache')
            generated, reused = self.complete('disk-hit', prompt)
            after = self.stats('after-disk-hit')
            ref = json.loads(ref_file.read_text())
            require(generated == ref['tokens'] and reused >= ref['shared_hit'] > 0 and
                    after['host_block_restores'] > before['host_block_restores'] and
                    after['host_image_restores'] > before['host_image_restores'], 'disk restore differs from warm run')
            return
        self.cold()
        text, _ = self.chat('text', 'Reply with exactly the uppercase word ORANGE.')
        require('ORANGE' in text.upper(), f'Qwen text response: {text!r}')
        self.image('vision', 'red')
        baseline, _ = self.complete('uncached', prompt, False)
        self.complete('original', original)
        warm, hit = self.complete('continuation', prompt)
        require(baseline == warm and 0 < hit <= len(original), 'Qwen continuation differs from cold')
        repeat, repeat_hit = self.complete('warm', prompt)
        require(repeat == baseline and repeat_hit >= hit, 'Qwen repeated continuation differs from cold')
        ref_file.write_text(json.dumps(dict(tokens=warm, shared_hit=hit, warm_hit=repeat_hit)) + '\n')
        self.stats('before-save')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['inkling', 'inkling-short', 'inkling-reload', 'qwen', 'qwen-restart'])
    parser.add_argument('--url', required=True)
    parser.add_argument('--request-log', type=Path, help='private file passed to Serve --request-log-jsonl')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase.startswith('inkling') and args.request_log is None:
        parser.error('Inkling phases require --request-log to verify DSpark and cancellation')
    probe = Probe(args)
    if args.phase == 'inkling-reload':
        probe.cold()
        reference = json.loads((args.output / 'inkling/blue-reference.json').read_text())
        blue, hit = probe.image('blue-after-restart', 'blue', ' Note.' * 4300)
        require(hit == 0 and blue == reference, 'changed image differs from cold restart')
    elif args.phase.startswith('inkling'):
        probe.inkling(args.phase == 'inkling-short')
    else:
        probe.qwen(args.phase == 'qwen-restart')
    if args.phase.startswith('inkling'):
        probe.verify_drafter()
    probe.save('result', dict(phase=args.phase, status='pass'))
    print(args.phase, 'passed', flush=True)


if __name__ == '__main__':
    main()
