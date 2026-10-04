"""Exercise actual HTTP serialization against a loopback OpenAI-compatible mock."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from doctor_agent.llm import OpenAICompatibleAdviser
from doctor_agent.simulator import demo_model


class HTTPTests(unittest.TestCase):
    def test_real_http_transport(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                content = json.dumps({"action_id": "exertional_pain"})
                data = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                adviser = OpenAICompatibleAdviser(f"http://127.0.0.1:{server.server_port}/v1")
                chosen = adviser.choose({}, [], [demo_model().actions[2]])
                self.assertEqual(chosen, "exertional_pain")
                self.assertEqual(requests[0][0], "/v1/chat/completions")
                self.assertEqual(requests[0][1]["model"], "openai/gpt-oss-20b")
                self.assertEqual(adviser.failures, 0)
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
