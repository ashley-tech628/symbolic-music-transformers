"""Small architecture tests; explicitly skipped when PyTorch is unavailable."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class ModelTests(unittest.TestCase):
    def test_lm_shape_and_tied_weights(self):
        import torch
        from musicgen.models import RelTransformerLM
        m=RelTransformerLM(24,d=16,h=4,layers=1,ff=32,max_rel=8,drop=0).eval()
        self.assertEqual(tuple(m(torch.ones(2,5,dtype=torch.long)).shape),(2,5,24))
        self.assertIs(m.head.weight,m.emb.weight)
    def test_lm_cannot_see_future_tokens(self):
        import torch
        from musicgen.models import RelTransformerLM
        torch.manual_seed(1)
        m=RelTransformerLM(24,d=16,h=4,layers=1,ff=32,max_rel=8,drop=0).eval()
        a=torch.tensor([[1,2,3,4]]);b=torch.tensor([[1,2,9,10]])
        with torch.no_grad():torch.testing.assert_close(m(a)[:,:2],m(b)[:,:2])
    def test_harmonizer_output_shape(self):
        import torch
        from musicgen.models import Seq2SeqHarmonizer
        m=Seq2SeqHarmonizer(20,30,d=16,heads=4,e_layers=1,d_layers=1,ff=32,drop=0).eval()
        self.assertEqual(tuple(m(torch.ones(2,5,dtype=torch.long),torch.ones(2,4,dtype=torch.long)).shape),(2,4,30))


if __name__=='__main__':unittest.main()
