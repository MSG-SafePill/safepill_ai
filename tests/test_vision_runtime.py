import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_service.vision_runtime import (
    ImageTooLarge, VisionTimingMiddleware, request_id, save_upload,
)


class UploadTests(unittest.TestCase):
    def test_boundary_bytes_are_preserved(self):
        path = save_upload(io.BytesIO(b"image"), max_bytes=5)
        try:
            self.assertEqual(path.read_bytes(), b"image")
        finally:
            path.unlink()

    def test_oversize_and_empty_uploads_leave_no_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            real_tempfile = tempfile.NamedTemporaryFile
            with patch("python_service.vision_runtime.tempfile.NamedTemporaryFile",
                       side_effect=lambda **kw: real_tempfile(dir=directory, **kw)):
                with self.assertRaises(ImageTooLarge):
                    save_upload(io.BytesIO(b"abcdef"), max_bytes=5)
                with self.assertRaises(ValueError):
                    save_upload(io.BytesIO(b""))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_read_failure_removes_partial_file(self):
        stream = unittest.mock.Mock()
        stream.read.side_effect = [b"partial", OSError("read failed")]
        with tempfile.TemporaryDirectory() as directory:
            real_tempfile = tempfile.NamedTemporaryFile
            with patch("python_service.vision_runtime.tempfile.NamedTemporaryFile",
                       side_effect=lambda **kw: real_tempfile(dir=directory, **kw)):
                with self.assertRaises(OSError):
                    save_upload(stream)
            self.assertEqual(list(Path(directory).iterdir()), [])


class TimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_and_complete_response_are_preserved(self):
        sent = []
        observed_ids = []

        async def app(scope, receive, send):
            observed_ids.append(request_id.get())
            self.assertEqual((await receive())["body"], b"image")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"success":true}'})

        async def receive():
            return {"type": "http.request", "body": b"image", "more_body": False}

        async def send(message):
            sent.append(message)

        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            await VisionTimingMiddleware(app)(
                {"type": "http", "path": "/classify-pills", "headers": [(b"x-request-id", b"trace-1")]}, receive, send,
            )
        self.assertEqual(observed_ids, ["trace-1"])
        self.assertIn((b"x-request-id", b"trace-1"), sent[0]["headers"])
        self.assertEqual(sent[1]["body"], b'{"success":true}')
        self.assertIn("upload_bytes=5", logs.output[0])
        self.assertEqual(request_id.get(), "startup")

    async def test_context_is_restored_after_failure(self):
        async def fail(*args):
            raise RuntimeError("test")
        with self.assertRaises(RuntimeError):
            await VisionTimingMiddleware(fail)({"type": "http", "path": "/ocr", "headers": []}, None, None)
        self.assertEqual(request_id.get(), "startup")

    async def test_nonvision_routes_pass_through(self):
        app = unittest.mock.AsyncMock()
        scope = {"type": "http", "path": "/health"}
        await VisionTimingMiddleware(app)(scope, None, None)
        app.assert_awaited_once_with(scope, None, None)


if __name__ == "__main__":
    unittest.main()
