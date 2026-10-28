#!/usr/bin/env python3
"""Проверки готовых C++ программ в отдельных процессах; сторонних пакетов нет.

Запускается build_and_test.py ПОСЛЕ сборки. В отличие от test_all.py требует
бинарники. Python не реализует обучение и не подменяет C++ загрузчик.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
BIN = None


def configure_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')


class ModelCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='pgrnn-cli-')
        cls.root = Path(cls.temp.name)
        suffix = '.exe' if os.name == 'nt' else ''
        cls.training = BIN / ('pgr_neural_training_demo' + suffix)
        cls.inference = BIN / ('pgr_neural_model_demo' + suffix)
        cls.output = cls.root / 'reference_run'
        result = cls.command(cls.training, '--output', cls.output)
        if result.returncode:
            cls.temp.cleanup()
            raise RuntimeError(result.stdout + result.stderr)
        cls.model = cls.output / 'model.pgrnn'
        cls.report = json.loads((cls.output / 'run.json').read_text(encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def command(executable, *args):
        return subprocess.run([str(executable), *(str(x) for x in args)], cwd=ROOT,
                              capture_output=True, encoding='utf-8', errors='replace', timeout=30)

    @classmethod
    def infer(cls, path, *extra):
        return cls.command(cls.inference, '--input', path, '--tokens', '1,2',
                           '--compiler', '0', '--stage', '0', *extra)

    def test_new_process_matches_training_probe(self):
        result = self.infer(self.model)
        self.assertEqual(result.returncode, 0, result.stderr)
        prediction = json.loads(result.stdout)
        self.assertEqual(prediction['format_version'], 1)
        for key, expected in self.report['probe'].items():
            self.assertEqual(prediction[key], expected, key)
        self.assertTrue(self.report['weights_saved'])
        self.assertTrue(self.report['model_roundtrip_exact'])
        self.assertFalse(self.report['training_resume_supported'])

    def test_python_independent_binary_reader(self):
        data = self.model.read_bytes()
        self.assertEqual(data[:8], b'PGRNN\r\n\x1a')
        self.assertEqual(struct.unpack_from('<4I', data, 8), (1, 128, 1, 0))
        self.assertEqual(struct.unpack_from('<Q', data, 24)[0], len(data))
        dims = struct.unpack_from('<7Q', data, 32)
        self.assertEqual(dims, (5, 4, 2, 2, 8, 2, 16))
        count, payload = struct.unpack_from('<2Q', data, 88)
        v, d, c, s, h, k, _ = dims
        self.assertEqual(count, v*d+h*(d+c+s)+h+k*h+k)
        self.assertEqual(payload, count*8)
        self.assertEqual(len(data), 128+payload+4)
        self.assertEqual(struct.unpack_from('<I', data, 104)[0], 1)
        self.assertEqual(data[108:128], bytes(20))
        self.assertEqual(zlib.crc32(data[:-4]), struct.unpack_from('<I', data, len(data)-4)[0])
        self.assertTrue(all(math.isfinite(x) for x in struct.unpack_from('<' + str(count) + 'd', data, 128)))

    def test_repeatable_saved_weights_and_reports(self):
        target = self.root / 'repeated_run'
        result = self.command(self.training, '--output', target)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('model.pgrnn', 'history.csv', 'run.json'):
            self.assertEqual((target/name).read_bytes(), (self.output/name).read_bytes(), name)

    def test_unicode_spaces_training_and_copy(self):
        target = self.root / 'Опыт с пробелами'
        result = self.command(self.training, '--epochs', '1', '--output', target)
        self.assertEqual(result.returncode, 0, result.stderr)
        copied = target / 'модель копия.pgrnn'
        result = self.infer(target/'model.pgrnn', '--copy-to', copied)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(copied.read_bytes(), (target/'model.pgrnn').read_bytes())

    def test_copy_never_overwrites_existing_file(self):
        target = self.root / 'occupied.pgrnn'
        target.write_bytes(b'existing user data\x00\xff')
        before = target.read_bytes()
        result = self.infer(self.model, '--copy-to', target)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(list(self.root.glob('.pgrnn-*.tmp')))

    def test_competing_processes_only_one_writer(self):
        target = self.root / 'competing.pgrnn'
        # Потоки нужны лишь тесту для одновременного запуска восьми процессов.
        # Сама сеть и её обучающий цикл по-прежнему работают в одном потоке.
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.infer(self.model, '--copy-to', target), range(8)))
        self.assertEqual(sum(result.returncode == 0 for result in results), 1)
        self.assertTrue(all(result.returncode in (0, 1) for result in results))
        self.assertEqual(target.read_bytes(), self.model.read_bytes())
        self.assertEqual(self.infer(target).returncode, 0)
        self.assertFalse(list(self.root.glob('.pgrnn-*.tmp')))

    def test_invalid_models_rejected_without_output(self):
        original = self.model.read_bytes()
        corrupted = bytearray(original)
        corrupted[128] ^= 1
        nan = bytearray(original)
        struct.pack_into('<Q', nan, 128, 0x7ff8000000000001)
        struct.pack_into('<I', nan, len(nan)-4, zlib.crc32(nan[:-4]))
        for index, data in enumerate((b'', original[:-1], original+b'x', bytes(corrupted), bytes(nan))):
            with self.subTest(index=index):
                source = self.root / ('bad_' + str(index) + '.pgrnn')
                source.write_bytes(data)
                target = self.root / ('bad_copy_' + str(index) + '.pgrnn')
                result = self.infer(source, '--copy-to', target)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('ERROR:', result.stderr)
                self.assertFalse(target.exists())

    def test_invalid_inference_arguments_rejected(self):
        base = ['--input', self.model, '--tokens', '1,2', '--compiler', '0', '--stage', '0']
        variants = [[], ['--help-bad'], base+['--stage', '1'], base+['--input', ''],
                    ['--input', self.model, '--tokens', '1,2'], base+['--copy-to']]
        for token in ('', '-1', '1,', ',1', '1,,2', '99999', '18446744073709551616'):
            variants.append(['--input', self.model, '--tokens', token, '--compiler', '0', '--stage', '0'])
        for key in ('--compiler', '--stage'):
            args = list(base)
            args[args.index(key)+1] = '99999'
            variants.append(args)
        before = self.model.read_bytes()
        for args in variants:
            with self.subTest(args=args):
                result = self.command(self.inference, *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('ERROR:', result.stderr)
        self.assertEqual(self.model.read_bytes(), before)
        self.assertEqual(self.command(self.inference, '--help').returncode, 0)

    def test_training_existing_directory_preserved(self):
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        result = self.command(self.training, '--output', self.output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)

    def test_invalid_training_arguments_create_nothing(self):
        variants = [('--epochs', '0'), ('--batch-size', '0'), ('--learning-rate', 'nan'),
                    ('--learning-rate', 'inf'), ('--learning-rate', '-1'), ('--seed', '-2'),
                    ('--epochs', '1x'), ('--seed', '18446744073709551616'),
                    ('--epochs', '1', '--epochs', '2')]
        for index, args in enumerate(variants):
            with self.subTest(args=args):
                target = self.root / ('invalid_training_' + str(index))
                result = self.command(self.training, *args, '--output', target)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(target.exists())
        self.assertEqual(self.command(self.training, '--help').returncode, 0)

    def test_read_only_inference(self):
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        result = self.infer(self.model)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)


if __name__ == '__main__':
    configure_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    args = parser.parse_args()
    BIN = args.bin_dir.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ModelCliTests)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
