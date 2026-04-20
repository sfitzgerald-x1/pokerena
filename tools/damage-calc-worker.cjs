#!/usr/bin/env node

const fs = require('fs');
const net = require('net');
const path = require('path');

const {calculateDamage, calculateDamageBatch, classifyMoveSupport} = require('./damage-calc-lib.cjs');
const WORKER_PROTOCOL_VERSION = 'pokerena.calc-worker.v1';

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = argv.slice(2);
  let socketPath = null;
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === '--socket') {
      socketPath = args[index + 1] || null;
      index += 1;
    }
  }
  if (!socketPath) {
    fail('Expected --socket PATH.');
  }
  return {socketPath};
}

function handleRequest(request) {
  if (!request || typeof request !== 'object' || Array.isArray(request)) {
    throw new Error('Worker request must be a JSON object.');
  }
  if (request.command === 'ping') {
    return {
      ok: true,
      result: {
        pong: true,
        protocol_version: WORKER_PROTOCOL_VERSION,
        commands: ['ping', 'damage', 'damage-batch', 'classify-move'],
      },
    };
  }
  if (request.command === 'damage') {
    return {ok: true, result: calculateDamage(request.payload)};
  }
  if (request.command === 'damage-batch') {
    return {ok: true, result: calculateDamageBatch(request.payload)};
  }
  if (request.command === 'classify-move') {
    return {ok: true, result: classifyMoveSupport(request.payload)};
  }
  throw new Error(`Unsupported worker command: ${String(request.command)}`);
}

function main() {
  const {socketPath} = parseArgs(process.argv);
  fs.mkdirSync(path.dirname(socketPath), {recursive: true});
  if (fs.existsSync(socketPath)) {
    fs.unlinkSync(socketPath);
  }

  const server = net.createServer(connection => {
    let buffer = '';
    connection.setEncoding('utf8');
    connection.on('data', chunk => {
      buffer += chunk;
    });
    connection.on('end', () => {
      let response;
      try {
        const request = JSON.parse(buffer.trim() || '{}');
        response = handleRequest(request);
      } catch (error) {
        response = {
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        };
      }
      connection.end(`${JSON.stringify(response)}\n`);
    });
  });

  const cleanup = () => {
    server.close(() => {
      if (fs.existsSync(socketPath)) {
        fs.unlinkSync(socketPath);
      }
      process.exit(0);
    });
  };

  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);
  process.on('exit', () => {
    if (fs.existsSync(socketPath)) {
      fs.unlinkSync(socketPath);
    }
  });

  server.listen(socketPath, () => {
    process.stdout.write(`worker listening on ${socketPath}\n`);
  });
}

main();
