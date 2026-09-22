import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pymupdf

from src.api.app import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        existing = self.root / 'old-index'
        existing.mkdir()
        (existing / 'manifest.json').write_text('{"vectors":2}')
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'rag_index_dir':'old-index','documents_dir':'documents'}))
        self.app = create_app(self.config)
        self.client = TestClient(self.app)

    def test_existing_and_model_override(self):
        original = self.config.read_bytes()
        self.assertEqual(self.client.get('/documents').json()['documents'][0]['document_id'],'existing')
        answer=dict(question='Why?',answer='Because.',sources=[],warnings=[],model_requested='other',model_returned='other',finish_reason='stop',usage={})
        with patch('src.api.app.ask',return_value=answer) as ask:
            response=self.client.post('/ask',json={'question':'Why?','document_id':'existing','model':'other','top_k':2})
            self.assertEqual(response.status_code,200)
            self.assertEqual(ask.call_args.kwargs['model'],'other')
            self.assertEqual(ask.call_args.kwargs['index_dir'],self.root/'old-index')
        self.assertEqual(original,self.config.read_bytes())

    def test_validation_and_busy(self):
        self.assertEqual(self.client.post('/ask',json={'question':'Why?','document_id':'../../etc'}).status_code,404)
        self.assertEqual(self.client.post('/ask',json={'question':' ','document_id':'existing'}).status_code,422)
        self.assertEqual(self.client.post('/ask',json={'question':'Why?','document_id':'existing','top_k':0}).status_code,422)
        self.app.state.processing_lock.acquire()
        try:
            self.assertEqual(self.client.post('/ask',json={'question':'Why?','document_id':'existing'}).status_code,409)
            self.assertEqual(self.client.get('/documents').status_code,200)
        finally:
            self.app.state.processing_lock.release()

    def test_upload_real_extraction_and_chunking(self):
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72,72),'A sample book page.')
            data=pdf.tobytes()
        def embed(chunks,directory,config):
            rows=[json.loads(line) for line in chunks.read_text().splitlines()]
            self.assertIn('A sample book page.',rows[0]['text'])
            directory.mkdir()
        def index(embeddings,directory):
            directory.mkdir()
            (directory/'manifest.json').write_text('{"vectors":1}')
            return {'vectors':1}
        with patch('src.documents.service.embed_chunks',side_effect=embed),patch('src.documents.service.build_index',side_effect=index):
            response=self.client.post('/documents',files={'file':('../sample.pdf',data,'application/pdf')})
        self.assertEqual(response.status_code,201,response.text)
        result=response.json()
        self.assertEqual(result['name'],'sample.pdf')
        self.assertEqual(result['pages'],1)
        self.assertTrue((self.root/'documents'/result['document_id']/'raw/sample.pdf').is_file())
        self.assertEqual(len(self.client.get('/documents').json()['documents']),2)

    def test_failed_preparation_hidden_and_lock_released(self):
        response=self.client.post('/documents',files={'file':('bad.pdf',b'not pdf','application/pdf')})
        self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(len(self.client.get('/documents').json()['documents']),1)
        self.assertFalse(self.app.state.processing_lock.locked())
        with patch('src.api.app.ask',side_effect=RuntimeError('Model unavailable')):
            self.assertEqual(self.client.post('/ask',json={'question':'Why?','document_id':'existing'}).status_code,503)
        self.assertFalse(self.app.state.processing_lock.locked())


if __name__ == '__main__':
    unittest.main()
