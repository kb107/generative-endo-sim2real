"""Qwen size handling and the Gemini client's dry-run, cap, retry, size checks (no network, no key)."""
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from generative_sim2real.foundation import gemini_edit as ge
from generative_sim2real.foundation import items, qwen_edit


def test_qwen_fit_to_size_records_resize():
    same, rec = qwen_edit.fit_to_size(Image.new('RGB', (512, 512)))
    assert same.size == (512, 512) and rec == {'model_output_size': [512, 512], 'resized_back': False, 'resample': None}
    big, rec = qwen_edit.fit_to_size(Image.new('RGB', (1024, 1024)))
    assert big.size == (512, 512) and rec['resized_back'] and rec['model_output_size'] == [1024, 1024]
    with pytest.raises(ValueError):
        qwen_edit.fit_to_size(Image.new('RGB', (1024, 768)))


def write_inputs(tmp_path, n=3):
    for name in ['ref1', 'ref2'] + [f'r{i}' for i in range(n)]:
        Image.new('RGB', (512, 512), (10, 20, 30)).save(tmp_path / f'{name}.png')
    spec = {'fixed8': [{'cond': i, 'sample_id': f's{i}', 'real': str(tmp_path / f'r{i}.png')} for i in range(n)],
            'refs': [{'file': str(tmp_path / 'ref1.png')}, {'file': str(tmp_path / 'ref2.png')}]}
    path = tmp_path / 'inputs.json'
    path.write_text(json.dumps(spec))
    return path


def test_load_items(tmp_path):
    got, refs = items.load_items(write_inputs(tmp_path), 'fixed8')
    assert [g['name'] for g in got] == ['c0', 'c1', 'c2'] and len(refs) == 2


def test_gemini_cost_estimate_uses_verified_prices():
    cost = ge.estimate_cost('gemini-nano-banana-2.1', '1K', 8)
    assert 0.0336 < cost['per_request_usd'] < 0.05 and cost['total_usd'] == pytest.approx(8 * cost['per_request_usd'])
    with pytest.raises(ValueError):
        ge.estimate_cost('unknown-model', '1K', 1)


def test_gemini_dry_run_makes_no_client_and_needs_no_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    genai = pytest.importorskip("google.genai")

    def boom(*a, **k):
        raise AssertionError('network client constructed in dry run')
    monkeypatch.setattr(genai, 'Client', boom)
    code = ge.main(['--set', 'fixed8', '--inputs', str(write_inputs(tmp_path)), '--out', str(tmp_path / 'o')])
    out = capsys.readouterr().out
    assert code == 0 and 'DRY RUN' in out and 'estimated cost' in out
    assert not (tmp_path / 'o').exists()


def test_gemini_execute_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with pytest.raises(SystemExit):
        ge.main(['--set', 'fixed8', '--inputs', str(write_inputs(tmp_path)), '--out', str(tmp_path / 'o'), '--execute'])


def test_budget_hard_cap():
    budget = ge.Budget(2)
    budget.take(), budget.take()
    with pytest.raises(RuntimeError):
        budget.take()


class FakeInteractions:
    def __init__(self, fail_times=0, size=(1024, 1024), message='503 UNAVAILABLE'):
        self.calls, self.fail_times, self.size, self.message = 0, fail_times, size, message

    def create(self, **kwargs):
        import base64
        import io
        self.calls += 1
        self.kwargs = kwargs
        if self.calls <= self.fail_times:
            raise RuntimeError(self.message)
        buf = io.BytesIO()
        Image.new('RGB', self.size, (1, 2, 3)).save(buf, 'PNG')
        return SimpleNamespace(output_image=SimpleNamespace(data=base64.b64encode(buf.getvalue()).decode()), usage=None)


def test_run_one_retries_transient_errors_and_counts_every_attempt(tmp_path):
    paths = [str(tmp_path / 'x.png')] * 3
    Image.new('RGB', (8, 8)).save(paths[0])
    client = SimpleNamespace(interactions=FakeInteractions(fail_times=2))
    budget = ge.Budget(5)
    (image, _), attempt = ge.run_one(client, budget, 'm', 'interactions', paths, '1K', retries=2, sleep=lambda s: None)
    assert attempt == 2 and budget.used == 3 and image.size == (1024, 1024)
    req = client.interactions.kwargs
    assert req['input'][0]['type'] == 'text' and len(req['input']) == 4
    assert req['response_format'] == {'type': 'image', 'aspect_ratio': '1:1', 'image_size': '1K'}


def test_run_one_does_not_retry_permanent_errors_and_respects_cap(tmp_path):
    paths = [str(tmp_path / 'x.png')]
    Image.new('RGB', (8, 8)).save(paths[0])
    client = SimpleNamespace(interactions=FakeInteractions(fail_times=9, message='400 INVALID_ARGUMENT'))
    with pytest.raises(RuntimeError):
        ge.run_one(client, ge.Budget(5), 'm', 'interactions', paths, '1K', retries=2, sleep=lambda s: None)
    assert client.interactions.calls == 1
    flaky = SimpleNamespace(interactions=FakeInteractions(fail_times=9))
    with pytest.raises(RuntimeError, match='cap'):
        ge.run_one(flaky, ge.Budget(2), 'm', 'interactions', paths, '1K', retries=5, sleep=lambda s: None)
    assert flaky.interactions.calls == 2


def test_check_output_aspect_and_resize():
    image, rec = ge.check_output(Image.new('RGB', (1024, 1024)))
    assert image.size == (512, 512) and rec['resized_back'] and rec['aspect_ok']
    with pytest.raises(ValueError, match='aspect'):
        ge.check_output(Image.new('RGB', (1024, 768)))


def test_execute_logs_per_image_json_without_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('GEMINI_API_KEY', 'SECRET-KEY-123')
    genai = pytest.importorskip("google.genai")
    fake = FakeInteractions()
    monkeypatch.setattr(genai, 'Client', lambda api_key: SimpleNamespace(interactions=fake))
    out = tmp_path / 'o'
    code = ge.main(['--set', 'fixed8', '--inputs', str(write_inputs(tmp_path)), '--out', str(out),
                    '--execute', '--max-requests', '2'])
    assert code == 0 and fake.calls == 2  # cap 2 of 3 images
    log = json.loads((out / 'c0.json').read_text())
    assert log['status'] == 'ok' and log['resized_back'] and log['model'] == ge.DEFAULT_MODEL
    assert 'SECRET-KEY-123' not in capsys.readouterr().out + (out / 'c0.json').read_text()
    assert Image.open(out / 'c1.png').size == (512, 512) and not (out / 'c2.png').exists()


def test_contact_sheet_layout_and_missing_editor_cell(tmp_path):
    from generative_sim2real.foundation import contact_sheet
    spec = json.loads(write_inputs(tmp_path).read_text())
    for row in spec['fixed8']:
        row.update(control=row['real'], target_correct=row['real'])
    out = tmp_path / 'sheet.png'
    contact_sheet.build(spec, [('Qwen', str(tmp_path / 'missing_{cond}.png'))], out, cell=32)
    sheet = Image.open(out)
    assert sheet.size == (32 * 6, 22 + 32 * 3)  # 3 fixed columns + 1 editor + 2 refs; 3 rows
