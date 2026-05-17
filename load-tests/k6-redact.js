import http from "k6/http";
import { check } from "k6";

export const options = {
  scenarios: {
    pii_redaction: {
      executor: "constant-arrival-rate",
      rate: 100,
      timeUnit: "1s",
      duration: "2m",
      preAllocatedVUs: 50,
      maxVUs: 200,
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<500"],
  },
};

const payload = JSON.stringify({
  text: "Call Maya Chen at +1 (415) 555-0124 or email maya.chen@example.com. API key: sk-test-1234567890abcdef.",
});

export default function () {
  const response = http.post("http://localhost:8080/redact", payload, {
    headers: { "content-type": "application/json" },
  });

  check(response, {
    "status is 200": (r) => r.status === 200,
    "redacts email": (r) => r.body.includes("PRIVATE_EMAIL"),
    "redacts phone": (r) => r.body.includes("PRIVATE_PHONE"),
  });
}
