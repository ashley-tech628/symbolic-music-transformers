import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from musicgen.history import extract
from musicgen.midi import read_midi, summarize

ROOT=Path(__file__).resolve().parent.parent


def midi(track, division=96):
    return b'MThd'+struct.pack('>IHHH',6,0,1,division)+b'MTrk'+struct.pack('>I',len(track))+track


class MidiTests(unittest.TestCase):
    def parse(self, data):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.mid';p.write_bytes(data)
            return read_midi(p)
    def test_running_status_and_velocity_zero(self):
        notes=self.parse(midi(bytes([0,0x90,60,100,96,60,0,0,0xff,0x2f,0])))
        self.assertEqual(len(notes),1)
        self.assertAlmostEqual(notes[0].end,0.5)
        self.assertEqual(notes[0].pitch,60)
    def test_tempo_change_affects_elapsed_time(self):
        tr=bytes([0,0x90,60,100,96,0xff,0x51,3,0x0f,0x42,0x40,96,0x80,60,0,0,0xff,0x2f,0])
        self.assertAlmostEqual(self.parse(midi(tr))[0].end,1.5)
    def test_truncated_track_rejected(self):
        with self.assertRaises(ValueError):self.parse(midi(bytes([0,0x90,60,100]))[:-1])
    def test_smpte_rejected(self):
        with self.assertRaisesRegex(ValueError,'SMPTE'):self.parse(midi(b'',0xe728))
    def test_unterminated_note_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unterminated'):
            self.parse(midi(bytes([0,0x90,60,100,0,0xff,0x2f,0])))
    def test_bundled_samples_have_four_note_tracks(self):
        for f in (ROOT/'examples').glob('*.mid'):
            notes=read_midi(f)
            self.assertEqual(summarize(notes)['tracks_with_notes'],4)
            self.assertGreater(len(notes),50)
            self.assertTrue(all(0<=n.pitch<=127 and 0<=n.start<n.end for n in notes))


class EvidenceTests(unittest.TestCase):
    def test_summary_matches_preserved_output(self):
        saved=json.loads((ROOT/'results/historical/summary.json').read_text(encoding='utf-8'))
        self.assertEqual(extract(ROOT),saved)
    def test_historical_notebook_hash_matches_source(self):
        manifest=json.loads((ROOT/'docs/source-manifest.json').read_text(encoding='utf-8'))
        entry=next(x for x in manifest['files'] if x['source']=='assignment2/assignment2.ipynb')
        actual=hashlib.sha256((ROOT/'results/historical/source_notebook.ipynb').read_bytes()).hexdigest()
        self.assertEqual(actual,entry['sha256'])
    def test_midi_copies_match_original_bytes(self):
        manifest=json.loads((ROOT/'docs/source-manifest.json').read_text(encoding='utf-8'))
        for f in (ROOT/'examples').glob('*.mid'):
            entry=next(x for x in manifest['files'] if x['source']=='assignment2/'+f.name)
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),entry['sha256'])


if __name__=='__main__':unittest.main()
