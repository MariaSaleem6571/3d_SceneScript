import os
from unittest import TestCase

from three_d_scene_script.gt_processor import SceneScriptProcessor


class GroundTruthProcessorTestCase(TestCase):
    processor: SceneScriptProcessor

    @classmethod
    def setUpClass(cls):
        cwd = os.getcwd()
        cls.processor = SceneScriptProcessor(os.path.join(cwd, '../0/ase_scene_language.txt'))

    def test_process(self):
        decoder_input_embeddings, gt_output_embeddings = self.processor.process()
        self.assertTrue(decoder_input_embeddings.shape[1:] == (16, 11))
        self.assertTrue(gt_output_embeddings.shape[1:] == (16, 11))

    def test_decode(self):
        pass
