import autocannon from "autocannon";

const instance = autocannon({
  url: "http://localhost:8080/redact",
  method: "POST",
  connections: 64,
  duration: 60,
  headers: {
    "content-type": "application/json",
  },
  body: JSON.stringify({
    text: "Email Alice Smith at alice@example.com, call +1 212 555 0199, and rotate sk-test-1234567890abcdef.",
  }),
});

autocannon.track(instance, { renderProgressBar: true });

instance.on("done", (result) => {
  console.log(autocannon.printResult(result));
});
