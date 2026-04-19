#!/usr/bin/env node

const {calculateDamage, fail} = require('./damage-calc-lib.cjs');

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

async function main() {
  const rawInput = (await readStdin()).trim();
  if (!rawInput) {
    fail('Expected a JSON payload on stdin.');
  }

  let payload;
  try {
    payload = JSON.parse(rawInput);
  } catch (error) {
    fail(`Failed to parse JSON input: ${error.message}`);
  }

  const response = calculateDamage(payload);

  process.stdout.write(`${JSON.stringify(response)}\n`);
}

main().catch(error => {
  fail(error instanceof Error ? error.message : String(error));
});
